import logging
import numpy as np
import dolfin as df
import solver_base as sb
from finmag.util.timings import timings, mtimed
import finmag.util.solver_benchmark as bench


from finmag.native.treecode_bem import FastSum
from finmag.native.treecode_bem import compute_solid_angle_single


def GetDet3(x,y,z):
    """
    helper function
    """
    d = x[0] * y[1] * z[2] + x[1] * y[2] * z[0] \
      + x[2] * y[0] * z[1] - x[0] * y[2] * z[1] \
      - x[1] * y[0] * z[2] - x[2] * y[1] * z[0];
    return d;

def GetTetVol(x1,x2,x3,x4):
    """
    helper fuctioen
    """
    v = GetDet3(x2, x3, x4) - GetDet3(x1, x3, x4) + GetDet3(x1, x2, x4) - GetDet3(x1, x2, x3);
    return 1.0 / 6.0 * v;


def compute_bnd_mapping(mesh):
    """
    we can remove this function if the order bug of dolfin is fixed
    """
    mesh.init()

    number_nodes=mesh.num_vertices()

    number_faces_bnd=0

    bnd_face_verts=[]
    gnodes_to_bnodes=np.zeros(number_nodes,int)
    node_at_boundary=np.zeros(number_nodes,int)
    nodes_xyz=mesh.coordinates()

    for face in df.faces(mesh):
        cells = face.entities(3)
        if len(cells)==1:
            face_nodes=face.entities(0)
            cell = df.Cell(mesh,cells[0])
            cell_nodes=cell.entities(0)

            #print set(cell_nodes)-set(face_nodes),face_nodes
            tmp_set=set(cell_nodes)-set(face_nodes)

            x1=nodes_xyz[tmp_set.pop()]
            x2=nodes_xyz[face_nodes[0]]
            x3=nodes_xyz[face_nodes[1]]
            x4=nodes_xyz[face_nodes[2]]

            tmp_vol=GetTetVol(x1,x2,x3,x4)

            local_nodes=[face_nodes[0]]
            if tmp_vol<0:
                local_nodes.append(face_nodes[2])
                local_nodes.append(face_nodes[1])
            else:
                local_nodes.append(face_nodes[1])
                local_nodes.append(face_nodes[2])

            bnd_face_verts.append(local_nodes)
            for i in face_nodes:
                node_at_boundary[i]=1
            number_faces_bnd+=1

    bnd_face_verts=np.array(bnd_face_verts)

    number_nodes_bnd=0
    for i in range(number_nodes):
        if node_at_boundary[i]>0:
            gnodes_to_bnodes[i]=number_nodes_bnd
            number_nodes_bnd+=1
        else:
            gnodes_to_bnodes[i]=-1

    return (bnd_face_verts,gnodes_to_bnodes,number_faces_bnd,number_nodes_bnd)

logger = logging.getLogger(name='finmag')
__all__ = ["TreecodeBEM"]
class TreecodeBEM(sb.FemBemDeMagSolver):
    @mtimed
    def __init__(self,mesh,m, parameters=sb.default_parameters , degree=1, element="CG",
                 project_method='magpar', unit_length=1,Ms = 1.0,bench = False,
                 mac=0.3,p=3,num_limit=100,correct_factor=5):
        
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

        self.p=p
        self.mac=mac
        self.num_limit=num_limit
        self.correct_factor=correct_factor
        
        
        self.compute_triangle_normal()

        self.__bulid_Mapping()

        fast_sum=FastSum(p=self.p,mac=self.mac,num_limit=self.num_limit,r_eps=self.correct_factor)

        xt=self.mesh.coordinates()
        self.bnd_face_nodes=np.array(self.bnd_face_nodes,dtype=np.int32)
        self.g2b=np.array(self.gnodes_to_bnodes,dtype=np.int32)

        fast_sum.init_mesh(xt,self.t_normals,self.bnd_face_nodes,self.g2b,self.vert_bsa)
        self.fast_sum=fast_sum
        self.res=np.zeros(len(self.mesh.coordinates()))

    def __bulid_Mapping(self):
        self.bnd_face_nodes,\
        self.gnodes_to_bnodes,\
        self.bnd_faces_number,\
        self.bnd_nodes_number=compute_bnd_mapping(self.mesh)

        self.nodes_number=self.mesh.num_vertices()

        vert_bsa=np.zeros(self.nodes_number)

        g2b=self.gnodes_to_bnodes
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


        for i in range(self.nodes_number):
            j=g2b[i]
            if j<0:
                vert_bsa[i]=0
            else:
                vert_bsa[i]=vert_bsa[i]/(4*np.pi)-1

        self.vert_bsa=vert_bsa


    def compute_triangle_normal(self):

        self.face_nodes=[]
        self.face_norms=[]
        self.t_normals=[]
        
        for face in df.faces(self.mesh):
            t=face.normal()  #one must call normal() before entities(3),...
            cells = face.entities(3)
            if len(cells)==1:
                face_nodes=face.entities(0)
                self.face_nodes.append(face_nodes)
                self.face_norms.append(t)
                self.t_normals.append([t.x(),t.y(),t.z()])

        self.t_normals=np.array(self.t_normals)
        self.face_nodes_array=np.array(self.face_nodes,dtype=np.int32)


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
        timings.start_next(self.__class__.__name__, "Compute phi2 at boundary")
        
        self.fast_sum.fastsum(self.res,self.phi1.vector().array())
        #self.fast_sum.directsum(self.res,self.phi1.vector().array())

        #print 'phi2 at boundary',self.res
        self.phi2.vector().set_local(self.res)
        
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
    #mesh = df.UnitSphereMesh(n)
    #mesh=df.Mesh('tet.xml')
    mesh = df.BoxMesh(0, 0, 0, 100, 1, 1, 100, 1, 1)
    expr = df.Expression(('4.0*sin(x[0])', '4*cos(x[0])','0'))
    
    Vv=df.VectorFunctionSpace(mesh, "Lagrange", 1)

   

    Ms = 8.6e5
    expr = df.Expression(('cos(x[0])', 'sin(x[0])','0'))
    m = df.interpolate(expr, Vv)
    #m = df.interpolate(df.Constant((1, 0, 0)), Vv)
    
    from finmag.energies.demag.solver_fk import FemBemFKSolver as FKSolver
    
    fk = FKSolver(mesh, m, Ms=Ms)
    f1= fk.compute_field()


    demag=TreecodeBEM(mesh,m,mac=0.3,p=5,num_limit=100,correct_factor=5,Ms=Ms)
    f2=demag.compute_field()
    
    f3=f1-f2
    print f3/f1
    
