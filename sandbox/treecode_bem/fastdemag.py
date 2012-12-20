from dolfin import *
import dolfin as df
import numpy as np


import time
from fastsum_lib import FastSum
from fastsum_lib import compute_solid_angle_single
from fastsum_lib  import compute_boundary_element


def compute_cell_volume(mesh):
    V = df.FunctionSpace(mesh, 'DG', 0)
    v = df.TestFunction(V)
    tet_vol=df.assemble(v * df.dx)
    return tet_vol.array()


def compute_minus_node_volume_vector(mesh):
    V=VectorFunctionSpace(mesh, 'Lagrange', 1)
    v = df.TestFunction(V)
    node_vol= df.assemble(df.dot(v,
            df.Constant([-1,-1,-1])) * df.dx)
    return node_vol.array()

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
    we can remove thi function if the order bug of dolfin is fixed
    """
    mesh.init()

    number_nodes=mesh.num_vertices()

    number_nodes_bnd=0
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


class Demag():

    def __init__(self,p=6,mac=0.5,triangle_p=1,num_limit=100):
        self.p=p
        self.mac=mac
        self.triangle_p=triangle_p
        self.num_limit=num_limit
        self.in_jacobian=False


    def setup(self,Vv,m,Ms,unit_length=1):
        self.m=m
        self.Vv=Vv
        self.Ms=Ms
        self.mesh=Vv.mesh()
        self.V=FunctionSpace(self.mesh, 'Lagrange', 1)
        self.phi = Function(self.V)
        self.phi_charge = Function(self.V)
        self.phi1 = df.Function(self.V)
        self.phi2 = df.Function(self.V)
        self.laplace_zeros = df.Function(self.V).vector()


        self.field = Function(self.Vv)

        self.find_max_d()
        self.mesh.coordinates()[:]/=self.max_d

        u = TrialFunction(self.V)
        v = TestFunction(self.Vv)
        a = inner(grad(u), v)*dx
        self.G = df.assemble(a)

        self.L = compute_minus_node_volume_vector(self.mesh)

        self.compute_triangle_normal()

        self.__bulid_Mapping()
        self.__bulid_Matrix(debug=False)

        fast_sum=FastSum(p=self.p,mac=self.mac,num_limit=self.num_limit,triangle_p=self.triangle_p)

        xt=self.mesh.coordinates()
        #tet_nodes=np.array(self.mesh.cells(),dtype=np.int32)
        self.bnd_face_nodes=np.array(self.bnd_face_nodes,dtype=np.int32)
        self.g2b=np.array(self.gnodes_to_bnodes,dtype=np.int32)

        fast_sum.init_mesh(xt,self.t_normals,self.bnd_face_nodes,self.g2b,self.vert_bsa)
        self.fast_sum=fast_sum
        self.res=np.zeros(len(self.mesh.coordinates()))

        self.mesh.coordinates()[:]*=self.max_d


    def __bulid_Mapping(self):
        self.bnd_face_nodes,\
        self.gnodes_to_bnodes,\
        self.bnd_faces_number,\
        self.bnd_nodes_number=compute_bnd_mapping(self.mesh)

        #print 'trianle number',self.bnd_faces_number,self.bnd_nodes_number
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

    def __bulid_Matrix(self,debug=False):

        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        w = df.TrialFunction(self.Vv)


        b = self.Ms * df.inner(w, df.grad(v)) * df.dx
        self.D=df.assemble(b)

        a = df.inner(df.grad(u),df.grad(v))*df.dx
        self.K1=df.assemble(a)


        self.bc = df.DirichletBC(self.V, self.phi2,df.DomainBoundary())
        self.K2=self.K1.copy()
        self.bc.apply(self.K2)


        self.poisson_solver = df.KrylovSolver(self.K1)
        self.laplace_solver = df.KrylovSolver(self.K2)
        self.laplace_solver.parameters["preconditioner"]["same_nonzero_pattern"] = True


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


    def find_max_d(self):
        xt=self.mesh.coordinates()
        max_v=xt.max(axis=0)
        min_v=xt.min(axis=0)
        max_d=max(max_v-min_v)
        self.max_d=max_d

    #used for debug
    def get_B_length(self):
        return self.fast_sum.get_B_length(),self.bnd_nodes_number**2


    def compute_field(self):

        g1=self.D*self.m.vector()

        self.poisson_iter = self.poisson_solver.solve(self.phi1.vector(), g1)

        #print '='*100,'phi1\n',self.phi1.vector().array()

        self.fast_sum.fastsum(self.res,self.phi1.vector().array())

        #print 'phi2 at boundary',self.res

        self.phi2.vector().set_local(self.res)
        b = self.laplace_zeros
        self.bc.apply(b)
        self.laplace_iter = self.laplace_solver.solve(self.phi2.vector(), b)

        self.phi.vector()[:] = self.phi1.vector() \
                             + self.phi2.vector()

        demag_field = self.G * self.phi.vector()

        return demag_field.array()/self.L

if __name__ == "__main__":

    n=4
    #mesh = UnitCube(n, n, n)
    #mesh = Box(-1, 0, 0, 1, 1, 1, 10, 2, 2)
    mesh = UnitSphere(n)
    #mesh=df.Mesh('tet.xml')

    Vv = df.VectorFunctionSpace(mesh, 'Lagrange', 1)

    Ms = 8.6e5
    #expr = df.Expression(('cos(x[0])', 'sin(x[0])','0'))
    #m = interpolate(expr, Vv)
    m = interpolate(Constant((1, 0, 0)), Vv)


    demag=Demag(mac=0.4,p=6,triangle_p=1,num_limit=2)
    demag.setup(Vv,m,Ms)
    print demag.compute_field()

    #demag.fast_sum.free_memory()

    #cProfile.run('demag.compute_field();')

    #print len(mesh.coordinates())

    #print np.array(demag.compute_field())

    """
    p=np.array([0.0,0,0])
    x1=np.array([1.0,0,0])
    x2=np.array([0.0,1.0,0])
    x3=np.array([0.0,0,1.0])
    print compute_solid_angle(p,x1,x2,x3)
    print compute_solid_angle_single(p,x1,x2,x3)
    """
