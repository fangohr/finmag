import logging
import numpy as np
import dolfin as df
import solver_base as sb
from finmag.util.meshes import sphere
from finmag.util.timings import timings, mtimed
import finmag.util.solver_benchmark as bench


from finmag.native.treecode_bem import FastSum
from finmag.native.treecode_bem import compute_solid_angle_single

logger = logging.getLogger(name='finmag')
__all__ = ["TreecodeBEM"]
class TreecodeBEM(sb.FemBemDeMagSolver):
    def __init__(self,mesh,m, parameters=sb.default_parameters , degree=1, element="CG",
                 project_method='magpar', unit_length=1,Ms = 1.0,bench = False,
                 mac=0.3,p=3,num_limit=100,correct_factor=10, type_I=True):
        
        sb.FemBemDeMagSolver.__init__(self,mesh,m, parameters, degree, element=element,
                                      project_method = project_method,
                                      unit_length = unit_length,Ms = Ms,bench = bench)
        self.__name__ = "Treecode Demag Solver"
        
        
        #Linear Solver parameters
        method = parameters["poisson_solver"]["method"]
        pc = parameters["poisson_solver"]["preconditioner"]
        
        self.poisson_solver = df.KrylovSolver(self.poisson_matrix, method, pc)

        self.phi1 = df.Function(self.V)
        self.phi2 = df.Function(self.V)

        # Eq (1) and code-block 2 - two first lines.
        b = self.Ms*df.inner(self.w, df.grad(self.v))*df.dx
        self.D = df.assemble(b)
        
        self.mesh=mesh
        
        self.bmesh = df.BoundaryMesh(mesh,False)
        self.b2g_map = self.bmesh.vertex_map().array()
        
        self.compute_triangle_normal()

        self.__compute_bsa()
        
        fast_sum=FastSum(p=p,mac=mac,num_limit=num_limit,correct_factor=correct_factor,type_I=type_I)

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

        self.vert_bsa=vert_bsa[self.b2g_map]


    def compute_triangle_normal(self):

        self.t_normals=[]
        
        for face in df.faces(self.mesh):
            t=face.normal()  #one must call normal() before entities(3),...
            cells = face.entities(3)
            if len(cells)==1:
                self.t_normals.append([t.x(),t.y(),t.z()])

        self.t_normals=np.array(self.t_normals)
        


    #used for debug
    def get_B_length(self):
        return self.fast_sum.get_B_length(),self.bnd_nodes_number**2

    
    def solve(self):

        # Compute phi1 on the whole domain (code-block 1, last line)
        timings.start(self.__class__.__name__, "phi1 - matrix product")
        g1 = self.D*self.m.vector()

        timings.start_next(self.__class__.__name__, "phi1 - solve")
        if self.bench:
            bench.solve(self.poisson_matrix,self.phi1.vector(),g1, benchmark = True)
        else:
            timings.start_next(self.__class__.__name__, "1st linear solve")
            self.poisson_iter = self.poisson_solver.solve(self.phi1.vector(), g1)
            timings.stop(self.__class__.__name__, "1st linear solve")
        # Restrict phi1 to the boundary
        
        self.phi1_b = self.phi1.vector()[self.b2g_map]
        
        timings.start_next(self.__class__.__name__, "Compute phi2 at boundary")
        
        self.fast_sum.fastsum(self.phi2_b, self.phi1_b.array())
        #self.fast_sum.directsum(self.phi2_b,self.phi1_b.array())

        #print 'phi2 at boundary',self.res
        self.phi2.vector()[self.b2g_map[:]] = self.phi2_b
        
        # Compute Laplace's equation inside the domain,
        # eq. (2) and last code-block
        timings.start_next(self.__class__.__name__, "Compute phi2 inside")
        self.phi2 = self.solve_laplace_inside(self.phi2)

        # phi = phi1 + phi2, eq. (5)
        timings.start_next(self.__class__.__name__, "Add phi1 and phi2")
        self.phi.vector()[:] = self.phi1.vector() \
                             + self.phi2.vector()
        timings.stop(self.__class__.__name__, "Add phi1 and phi2")
        return self.phi

if __name__ == "__main__":

    n=4
    #mesh = UnitCube(n, n, n)
    #mesh = BoxMesh(-1, 0, 0, 1, 1, 1, 10, 2, 2)
    #mesh=sphere(3.0,0.3)
    #mesh=df.Mesh('tet.xml')
    #mesh = df.BoxMesh(0, 0, 0, 100, 1, 1, 100, 1, 1)
    #expr = df.Expression(('4.0*sin(x[0])', '4*cos(x[0])','0'))
    from finmag.util.meshes import elliptic_cylinder,sphere
    mesh = elliptic_cylinder(100,150,5,4.5,directory='meshes')
    
    Vv=df.VectorFunctionSpace(mesh, "Lagrange", 1)
    
    Ms = 8.6e5
    expr = df.Expression(('cos(x[0])', 'sin(x[0])','0'))
    m = df.interpolate(expr, Vv)
    m = df.interpolate(df.Constant((0, 0, 1)), Vv)
    
    from finmag.energies.demag.solver_fk import FemBemFKSolver as FKSolver
    
    fk = FKSolver(mesh, m, Ms=Ms)
    f1= fk.compute_field()


    demag=TreecodeBEM(mesh,m,mac=0.3,p=4,num_limit=100,correct_factor=10,Ms=Ms,type_I=True)
    f2=demag.compute_field()
    
    f3=f1-f2
    print f1[0:10],f2[0:10]
    print np.average(np.abs(f3[-200:]/f1[-200:]))
    
