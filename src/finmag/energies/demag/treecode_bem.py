import logging
import numpy as np
import dolfin as df
from aeon import default_timer
from finmag.util.consts import mu0
from finmag.util.meshes import nodal_volume
from finmag.native.treecode_bem import FastSum
from finmag.native.treecode_bem import compute_solid_angle_single
from finmag.util import helpers

from fk_demag import FKDemag

logger = logging.getLogger(name='finmag')

class TreecodeBEM(FKDemag):
    def __init__(self, mac=0.3, p=3, num_limit=100, correct_factor=10, 
                 type_I=True, name='Demag', Ts=None, thin_film=False):
        super(TreecodeBEM, self).__init__(name=name, Ts=Ts, thin_film=thin_film)
        
        self.mac = mac
        self.p = p
        self.num_limit = num_limit
        self.correct_factor = correct_factor
        self.type_I = type_I


    def setup(self, S3, m, Ms, unit_length=1):
        
        self.m = m
        self.Ms = Ms
        self.unit_length = unit_length

        mesh = S3.mesh()
        self.S1 = df.FunctionSpace(mesh, "Lagrange", 1)
        self.S3 = S3
        self.dim = mesh.topology().dim()

        self._test1 = df.TestFunction(self.S1)
        self._trial1 = df.TrialFunction(self.S1)
        self._test3 = df.TestFunction(self.S3)
        self._trial3 = df.TrialFunction(self.S3)

        # for computation of energy
        self._nodal_volumes = nodal_volume(self.S1, unit_length)
        self._H_func = df.Function(S3)  # we will copy field into this when we need the energy
        self._E_integrand = -0.5 * mu0 * df.dot(self._H_func, self.m * self.Ms)
        self._E = self._E_integrand * df.dx
        self._nodal_E = df.dot(self._E_integrand, self._test1) * df.dx
        self._nodal_E_func = df.Function(self.S1)

        # for computation of field and scalar magnetic potential
        self._poisson_matrix = self._poisson_matrix()
        self._poisson_solver = df.KrylovSolver(self._poisson_matrix.copy(),
            self.parameters['phi_1_solver'], self.parameters['phi_1_preconditioner'])
        self._poisson_solver.parameters.update(self.parameters['phi_1'])
        self._laplace_zeros = df.Function(self.S1).vector()
        self._laplace_solver = df.KrylovSolver(
                self.parameters['phi_2_solver'], self.parameters['phi_2_preconditioner'])
        self._laplace_solver.parameters.update(self.parameters['phi_2'])
        # We're setting 'same_nonzero_pattern=True' to enforce the
        # same matrix sparsity pattern across different demag solves,
        # which should speed up things.
        self._laplace_solver.parameters["preconditioner"]["structure"] = "same_nonzero_pattern"
        
        
        self._phi_1 = df.Function(self.S1)  # solution of inhomogeneous Neumann problem
        self._phi_2 = df.Function(self.S1)  # solution of Laplace equation inside domain
        self._phi = df.Function(self.S1)  # magnetic potential phi_1 + phi_2

        # To be applied to the vector field m as first step of computation of _phi_1.
        # This gives us div(M), which is equal to Laplace(_phi_1), equation
        # which is then solved using _poisson_solver.
        self._Ms_times_divergence = df.assemble(self.Ms * df.inner(self._trial3, df.grad(self._test1)) * df.dx)
        
        #we move the bounday condition here to avoid create a instance each time when compute the 
        #magnetic potential 
        self.boundary_condition = df.DirichletBC(self.S1, self._phi_2, df.DomainBoundary())
        self.boundary_condition.apply(self._poisson_matrix)

        self._setup_gradient_computation()

        self.mesh= S3.mesh()

        self.bmesh = df.BoundaryMesh(self.mesh, 'exterior', False)
        #self.b2g_map = self.bmesh.vertex_map().array()
        self._b2g_map = self.bmesh.entity_map(0).array()

        self.compute_triangle_normal()

        self.__compute_bsa()

        fast_sum=FastSum(p=self.p, mac=self.mac, num_limit=self.num_limit, correct_factor=self.correct_factor, type_I=self.type_I)

        coords=self.bmesh.coordinates()
        face_nodes=np.array(self.bmesh.cells(),dtype=np.int32)

        fast_sum.init_mesh(coords,self.t_normals,face_nodes,self.vert_bsa)
        self.fast_sum=fast_sum

        self.phi2_b = np.zeros(self.bmesh.num_vertices())


    def __compute_bsa(self):

        vert_bsa=np.zeros(self.mesh.num_vertices())

        mc=self.mesh.cells()
        xyz=self.mesh.coordinates()
        for i in range(self.mesh.num_cells()):
            for j in range(4):

                tmp_omega=compute_solid_angle_single(
                    xyz[mc[i][j]],
                    xyz[mc[i][(j+1)%4]],
                    xyz[mc[i][(j+2)%4]],
                    xyz[mc[i][(j+3)%4]])

                vert_bsa[mc[i][j]]+=tmp_omega

        vert_bsa=vert_bsa/(4*np.pi)-1

        self.vert_bsa=vert_bsa[self._b2g_map]


    def compute_triangle_normal(self):

        self.t_normals=[]

        for face in df.faces(self.mesh):
            t=face.normal()  #one must call normal() before entities(3),...
            cells = face.entities(3)
            if len(cells)==1:
                self.t_normals.append([t.x(),t.y(),t.z()])

        self.t_normals=np.array(self.t_normals)
    
    def _compute_magnetic_potential(self):
        # compute _phi_1 on the whole domain
        g_1 = self._Ms_times_divergence * self.m.vector()
        
        self._poisson_solver.solve(self._phi_1.vector(), g_1)

        # compute _phi_2 on the boundary using the Dirichlet boundary
        # conditions we get from BEM * _phi_1 on the boundary.
        
        phi_1 = self._phi_1.vector()[self._b2g_map]
        
        self.fast_sum.fastsum(self.phi2_b, phi_1.array())
        
        self._phi_2.vector()[self._b2g_map[:]] = self.phi2_b

        A = self._poisson_matrix
        b = self._laplace_zeros
        self.boundary_condition.set_value(self._phi_2)
        self.boundary_condition.apply(A,b)

        # compute _phi_2 on the whole domain
        self._laplace_solver.solve(A, self._phi_2.vector(), b)
        # add _phi_1 and _phi_2 to obtain magnetic potential
        self._phi.vector()[:] = self._phi_1.vector() + self._phi_2.vector()



def compare_field(f1,f2):
    f1.shape=(3,-1)
    f2.shape=(3,-1)
    d=f1-f2
    res=[]
    for i in range(d.shape[1]):
        v=f1[0][i]**2+f1[1][i]**2+f1[2][i]**2
        t=d[0][i]**2+d[1][i]**2+d[2][i]**2
        res.append(t/v)

    f1.shape=(-1,)
    f2.shape=(-1,)

    return np.max(np.sqrt(res))

if __name__ == "__main__":

    n=4
    #mesh = UnitCubeMesh(n, n, n)
    #mesh = BoxMesh(-1, 0, 0, 1, 1, 1, 10, 2, 2)
    #mesh=sphere(3.0,0.3)
    #mesh=df.Mesh('tet.xml')
    #
    #expr = df.Expression(('4.0*sin(x[0])', '4*cos(x[0])','0'))
    from finmag.util.meshes import elliptic_cylinder, sphere
    mesh = elliptic_cylinder(100,150,5,4.5,directory='meshes')
    #mesh=box(0,0,0,5,5,100,5,directory='meshes')
    #mesh = df.BoxMesh(0, 0, 0, 100, 2, 2, 400, 2, 2)
    mesh=sphere(15,1,directory='meshes')
    Vv=df.VectorFunctionSpace(mesh, "Lagrange", 1)

    Ms = 8.6e5
    expr = df.Expression(('cos(x[0])', 'sin(x[0])','0'))
    m = df.interpolate(expr, Vv)
    #m = df.interpolate(df.Constant((1, 0, 0)), Vv)

    from finmag.energies.demag.fk_demag import FKDemag

    import time

    fk = FKDemag()
    fk.setup(Vv, m, Ms, unit_length=1e-9)
    start=time.time()
    f1= fk.compute_field()
    stop=time.time()


    demag=TreecodeBEM(mac=0.4,p=5,num_limit=1,correct_factor=10,type_I=False)
    demag.setup(Vv, m, Ms, unit_length=1e-9)
    start2=time.time()
    f2=demag.compute_field()
    stop2=time.time()

    f3=f1-f2
    print f1[0:10],f2[0:10]
    print np.average(np.abs(f3[:200]/f1[:200]))

    print 'max errror:',compare_field(f1,f2)

    """
    print stop-start,stop2-start2

    from aeon import default_timer
    print default_timer.report(20)
    """

