from dolfin import *
import dolfin as df
import numpy as np

import time
from fastsum_lib import FastSum 
import cProfile



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



class Demag():
    
    def __init__(self,triangle_p=1,p=6,mac=0.5,num_limit=400):
	self.triangle_p=triangle_p
	self.p=p
	self.mac=mac
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

        self.__bulid_Matrix(debug=False)

        self.field = Function(self.Vv)

	self.find_max_d()
	self.mesh.coordinates()[:]/=self.max_d

        u = TrialFunction(self.V)
        v = TestFunction(self.Vv)
        a = inner(grad(u), v)*dx
        self.G = df.assemble(a)
        
        self.L = compute_minus_node_volume_vector(self.mesh)

	self.compute_triangle_normal()

       	
        fast_sum=FastSum(p=self.p,mac=self.mac,num_limit=self.num_limit,\
				 triangle_p=self.triangle_p)

        xt=self.mesh.coordinates()
        tet_nodes=np.array(self.mesh.cells(),dtype=np.int32)
        #fast_sum.init_mesh(xt,self.t_normals,self.face_nodes_array)
        self.fast_sum=fast_sum
	self.res=np.zeros(len(self.mesh.coordinates()))

	self.mesh.coordinates()[:]*=self.max_d


    def __bulid_Matrix(self,debug=False):
        
        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        w = df.TrialFunction(self.Vv)


        #=============================================
        """
        D * m = g1
        """
        b = self.Ms * df.inner(w, df.grad(v)) * df.dx
        self.D=df.assemble(b)
        if debug:
            print '='*100,'D\n',self.D.array()

        #=============================================
        """
        K1 * phi1 = g1
        Eq. (51) from Schrefl et al. 2007
        """
        a = df.inner(df.grad(u),df.grad(v))*df.dx
        self.K1=df.assemble(a)
        if debug:
            print '='*100,'K1\n',self.K1.array()

        #=============================================


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
	print self.t_normals


    def find_max_d(self):
	xt=self.mesh.coordinates()
	max_v=xt.max(axis=0)
	min_v=xt.min(axis=0)
        max_d=max(max_v-min_v)
	self.max_d=max_d


    def compute_field(self):
        
        m=self.m.vector().array()  
        
        #self.fast_sum.update_charge(m)
        
        #self.fast_sum.fastsum(self.res)
	
        #self.phi.vector().set_local(self.res)
        
        #self.phi.vector()[:]*=(self.Ms/(4*np.pi))
        
	#demag_field = self.G * self.phi.vector()
        
	#return demag_field.array()/self.L

if __name__ == "__main__":
   
    n=2
    #mesh = UnitCube(n, n, n)
    #mesh = Box(-1, 0, 0, 1, 1, 1, 10, 2, 2)
    mesh = UnitSphere(n)
    
    Vv = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
    
    Ms = 8.6e5
    expr = df.Expression(('cos(x[0])', 'sin(x[0])','0'))
    m = interpolate(expr, Vv)
    m = interpolate(Constant((1, 0, 0)), Vv)
    
    
    demag=Demag(triangle_p=1)
    demag.setup(Vv,m,Ms)
    #print demag.compute_field()
    #demag.fast_sum.free_memory()
    
    #cProfile.run('demag.compute_field();')
    
    #print len(mesh.coordinates())
    
    #print np.array(demag.compute_field())
    
    
    
    
