from dolfin import *
import dolfin as df
import numpy as np

import time
from fastsum_lib import FastSum 
import cProfile


_nodes  =(
(0.,),
(-0.5773502691896257,
 0.5773502691896257),
(-0.7745966692414834,
 0.,
 0.7745966692414834),
(-0.861136311594053,
 -0.3399810435848562,
 0.3399810435848562,
 0.861136311594053),
(-0.906179845938664,
 -0.5384693101056829,
 0.,
 0.5384693101056829,
 0.906179845938664))

_weights=(
(2.,),
(1.,
 1.),
(0.5555555555555553,
 0.888888888888889,
 0.5555555555555553),
(0.3478548451374539,
 0.6521451548625462,
 0.6521451548625462,
 0.3478548451374539),
(0.2369268850561887,
 0.4786286704993665,
 0.5688888888888889,
 0.4786286704993665,
 0.2369268850561887))
    

dunavant_x=(
     (0.333333333333333,),
     (0.666666666666667,0.166666666666667,0.166666666666667),
     (0.333333333333333,0.600000000000000,0.200000000000000,0.200000000000000),
     (0.108103018168070,0.445948490915965,0.445948490915965,0.816847572980459,0.091576213509771,0.091576213509771)
     )
 
dunavant_y=(
     (0.333333333333333,),
     (0.166666666666667,0.166666666666667,0.666666666666667),
     (0.333333333333333,0.200000000000000,0.200000000000000,0.600000000000000),
     (0.445948490915965,0.445948490915965,0.108103018168070,0.091576213509771,0.091576213509771,0.816847572980459)
     )
        
dunavant_w=(
    (1.0,),
    (0.333333333333333,0.333333333333333,0.333333333333333),
    (-0.562500000000000,0.520833333333333,0.520833333333333,0.520833333333333),
    (0.223381589678011,0.223381589678011,0.223381589678011,0.109951743655322,0.109951743655322,0.109951743655322)
    )

dunavant_n=[1,2,3,6]


tet_x=(
(0.25,),
(0.1381966011250110,0.5854101966249680,0.1381966011250110,0.1381966011250110)
)

tet_y=(
(0.25,),
(0.1381966011250110,0.1381966011250110,0.5854101966249680,0.1381966011250110)
)

tet_z=(
(0.25,),
(0.1381966011250110,0.1381966011250110,0.1381966011250110,0.5854101966249680)
)

tet_w=(
(1.0,),
(0.25,0.25,0.25,0.25)
)

def length(p1,p2):
	return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)

def length2(p1,p2):
	return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2

def G(r1,r2):
	r=length(r1,r2)
	
	if r<1e-12:
            print 'hahahahahahahhahahahah'
            
            return 0
		
	return 1.0/(r)
    


def compute_area(p1,p2,p3):
    a=length(p1,p2)
    b=length(p1,p3)
    c=length(p2,p3)
    s=(a+b+c)/2.0
    return np.sqrt(s*(s-a)*(s-b)*(s-c))


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

def compute_node_volume(mesh):
    V=FunctionSpace(mesh, 'Lagrange', 1)
    v = df.TestFunction(V)
    node_vol= df.assemble(v * df.dx)
    return node_vol.array()


def compute_node_area(mesh):
    V=FunctionSpace(mesh, 'Lagrange', 1)
    v = df.TestFunction(V)
    node_area = df.assemble(v * df.ds)
    tmp=node_area.array()
    print 'area is: ',sum(tmp)
    for i in range(len(tmp)):
	    if tmp[i]==0:
		    tmp[i]=1
		    
    return tmp


def compute_det(x1,y1,z1,x2,y2,z2,x3,y3,z3):
    
    J1=(x2-x1)*(z3-z1)-(x3-x1)*(z2-z1)
    J2=(x2-x1)*(y3-y1)-(x3-x1)*(y2-y1)
    J3=(y2-y1)*(z3-z1)-(y3-y1)*(z2-z1)
    
    return J1,J2,J3

    
def compute_correction_simplified(sa,sb,sc,p1,p2,p3):
    
    x1,y1,z1=p1
    x2,y2,z2=p2
    x3,y3,z3=p3
    
    J1,J2,J3=compute_det(x1,y1,z1,x2,y2,z2,x3,y3,z3)
    
    r1=np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    r2=np.sqrt((x3-x1)**2+(y3-y1)**2+(z3-z1)**2)
    r3=np.sqrt((x3-x2)**2+(y3-y2)**2+(z3-z2)**2)    

    fa=np.sqrt(J1*J1+J2*J2+J3*J3)
    
    fb=(sb-sc)*(r1-r2)/(2*r3*r3)
    fc=(sb+sc+2*sa)/(4.0*r3)+(r2*r2-r1*r1)*(sb-sc)/(4.0*r3**3)
    fd=np.log(r1+r2+r3)-np.log(r1+r2-r3)
   
    return fa*(fb+fc*fd)


class FastDemag():
    
    def __init__(self,Vv, m, Ms,triangle_p=1,tetrahedron_p=0,p=6,mac=0.5):
        self.m=m
        self.Vv=Vv
        self.Ms=Ms
	self.triangle_p=triangle_p
        self.tetrahedron_p=tetrahedron_p
        self.mesh=Vv.mesh()
        self.V=FunctionSpace(self.mesh, 'Lagrange', 1)
        self.phi = Function(self.V)
        self.phi_charge = Function(self.V)
        self.field = Function(self.Vv)

        u = TrialFunction(self.V)
        v = TestFunction(self.Vv)
        a = inner(grad(u), v)*dx
        self.G = df.assemble(a)
        
        self.L = compute_minus_node_volume_vector(self.mesh)
        
        
        #self.compute_gauss_coeff_triangle()
        #self.compute_gauss_coeff_tetrahedron()
        #self.compute_affine_transformation_surface()
        #self.compute_affine_transformation_volume()
        #self.nodes=np.array(self.s_nodes+self.v_nodes)
        #self.weights=np.array(self.s_weight+self.v_weight)
        #self.charges=np.array(self.s_charge+self.v_charge)
        
        self.compute_triangle_normal()
        fast_sum=FastSum(p=p,mac=mac,num_limit=500,triangle_p=triangle_p,tetrahedron_p=tetrahedron_p)
        xt=self.mesh.coordinates()
        tet_nodes=np.array(self.mesh.cells(),dtype=np.int32)
        fast_sum.init_mesh(xt,self.t_normals,self.face_nodes_array,tet_nodes)
        self.fast_sum=fast_sum
	self.res=np.zeros(len(self.mesh.coordinates()))

        
        
    def compute_gauss_coeff_triangle(self):
        n=self.triangle_p
        
        self.s_x=dunavant_x[n]
        self.s_y=dunavant_y[n]
        self.s_w=np.array(dunavant_w[n])/2.0
        
        print self.s_x,self.s_y
    
    def compute_gauss_coeff_tetrahedron(self):
        n=self.tetrahedron_p
        
        self.v_x=tet_x[n]
        self.v_y=tet_y[n]
        self.v_z=tet_z[n]
        self.v_w=np.array(tet_w[n])/6.0


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

    
    def compute_affine_transformation_surface(self):
	
        m=self.m.vector().array()
        
        m=m.reshape((-1,3),order='F')
        
        cs=self.mesh.coordinates()
	
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
        
        self.s_nodes=[]
	self.s_weight=[]
        self.s_charge=[]
        

        def compute_det_xy(x1,y1,z1,x2,y2,z2,x3,y3,z3):
           
            a = y2*z1 - y3*z1 - y1*z2 + y3*z2 + y1*z3 - y2*z3
            b = x2*z1 - x3*z1 - x1*z2 + x3*z2 + x1*z3 - x2*z3
            c = x2*y1 - x3*y1 - x1*y2 + x3*y2 + x1*y3 - x2*y3

            det=abs((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1))

            det*=np.sqrt((a*a+b*b)/(c*c)+1)

            return det
	
        
        for i in range(len(self.face_nodes)):
            f_c=self.face_nodes[i]

            x1,y1,z1=cs[f_c[0]]
            x2,y2,z2=cs[f_c[1]]
            x3,y3,z3=cs[f_c[2]]
    
            c11=x2-x1
            c12=x3-x1
            c21=y2-y1
            c22=y3-y1
            c31=z2-z1
            c32=z3-z1

            t=self.face_norms[i]
            
            if abs(t.z())>abs(t.x()) and abs(t.z())>abs(t.y()):
                det=compute_det_xy(x1,y1,z1,x2,y2,z2,x3,y3,z3)
            elif abs(t.y())>abs(t.x()):
                det=compute_det_xy(z1,x1,y1,z2,x2,y2,z3,x3,y3)
            else:
                det=compute_det_xy(y1,z1,x1,y2,z2,x2,y3,z3,x3)
                              
            sa=(m[f_c[0]][0]*t.x()+m[f_c[0]][1]*t.y()+m[f_c[0]][2]*t.z())
            sb=(m[f_c[1]][0]*t.x()+m[f_c[1]][1]*t.y()+m[f_c[1]][2]*t.z())
            sc=(m[f_c[2]][0]*t.x()+m[f_c[2]][1]*t.y()+m[f_c[2]][2]*t.z())
            
            
            for j in range(len(self.s_x)):
                x=c11*self.s_x[j]+c12*self.s_y[j]+x1
                y=c21*self.s_x[j]+c22*self.s_y[j]+y1
                z=c31*self.s_x[j]+c32*self.s_y[j]+z1
                 
                self.s_nodes.append([x,y,z])
                self.s_weight.append(det*self.s_w[j])
                self.s_charge.append(sa+(sb-sa)*self.s_x[j]+(sc-sa)*self.s_y[j])


        
    def compute_affine_transformation_volume(self):
        v = TestFunction(self.V)
        K = df.assemble(df.div(self.m) * v * df.dx)
        L = df.assemble(v * df.dx)
        rho = K.array()/L.array()
        
        cs=self.mesh.coordinates()
        m=self.m.vector().array()
        n=len(m)/3
        
        def compute_divergence(cell):
            i=cell.entities(0)
            x1,y1,z1=cs[i[1]]-cs[i[0]]
            x2,y2,z2=cs[i[2]]-cs[i[0]]
            x3,y3,z3=cs[i[3]]-cs[i[0]]
            m0 = np.array([m[i[0]],m[i[0]+n],m[i[0]+2*n]])
            m1 = np.array([m[i[1]],m[i[1]+n],m[i[1]+2*n]]) - m0
            m2 = np.array([m[i[2]],m[i[2]+n],m[i[2]+2*n]]) - m0
            m3 = np.array([m[i[3]],m[i[3]+n],m[i[3]+2*n]]) - m0
            a1 = [y3*z2 - y2*z3, -x3*z2 + x2*z3, x3*y2 - x2*y3]
            a2 = [-y3*z1 + y1*z3, x3*z1 - x1*z3, -x3*y1 + x1*y3]
            a3 = [y2*z1 - y1*z2, -x2*z1 + x1*z2, x2*y1 - x1*y2]
            v = x3*y2*z1 - x2*y3*z1 - x3*y1*z2 + x1*y3*z2 + x2*y1*z3 - x1*y2*z3

            tmp=0
            for j in range(3):
                tmp += a1[j]*m1[j]+a2[j]*m2[j]+a3[j]*m3[j]

            tmp=-1.0*tmp/v
            return tmp,abs(v)
        
        self.v_nodes=[]
	self.v_weight=[]
        self.v_charge=[]
	for cell in df.cells(self.mesh):
            i=cell.entities(0)
            rho,det=compute_divergence(cell)

            x0,y0,z0=cs[i[0]]
            c11,c12,c13=cs[i[1]]-cs[i[0]]
            c21,c22,c23=cs[i[2]]-cs[i[0]]
            c31,c32,c33=cs[i[3]]-cs[i[0]]
            
            for j in range(len(self.v_w)):
                x=c11*self.v_x[j]+c21*self.v_y[j]+c31*self.v_z[j]+x0
                y=c12*self.v_x[j]+c22*self.v_y[j]+c32*self.v_z[j]+y0
                z=c13*self.v_x[j]+c23*self.v_y[j]+c33*self.v_z[j]+z0

                self.v_charge.append(rho)
                self.v_nodes.append([x,y,z])
                self.v_weight.append(det*self.v_w[j])

    
    def sum_directly(self):
        cs=self.mesh.coordinates()
        m=len(cs)
        n=len(self.nodes)        
    
        res=np.zeros(m)
        for i in range(m):
            for j in range(n):
                res[i]+=G(cs[i],self.nodes[j])*self.weights[j]*self.charges[j]

        print 'directly',res
        self.phi.vector().set_local(res)
        
        

    def compute_field(self):
        
        m=self.m.vector().array()  
        
        self.fast_sum.update_charge(m)
        
        self.fast_sum.fastsum(self.res)
        #self.fast_sum.exactsum(res)
	
        self.fast_sum.compute_correction(m,self.res)
        
        self.phi.vector().set_local(self.res)
        
        self.phi.vector()[:]*=(self.Ms/(4*np.pi))
        
	demag_field = self.G * self.phi.vector()
        
	return demag_field.array()/self.L
	

class Demag():
    
    def __init__(self,triangle_p=1,tetrahedron_p=1,p=3,mac=0.3,num_limit=100):
        
	self.triangle_p=triangle_p
        self.tetrahedron_p=tetrahedron_p
	self.p=p
	self.mac=mac
	self.num_limit=num_limit
	self.in_jacobian=False
        
        
    def setup(self,Vv,m,Ms,unit_length=1):
	self.m=m
        self.Vv=Vv
        self.Ms=Ms
	self.mesh=Vv.mesh()
        self.find_max_d()
        self.mesh.coordinates()[:]/=self.max_d
        
        self.V=FunctionSpace(self.mesh, 'Lagrange', 1)
        self.phi = Function(self.V)
        self.phi_charge = Function(self.V)
        self.field = Function(self.Vv)

        u = TrialFunction(self.V)
        v = TestFunction(self.Vv)
        a = inner(grad(u), v)*dx
        self.G = df.assemble(a)
        
        self.L = compute_minus_node_volume_vector(self.mesh)

	self.compute_triangle_normal()
        fast_sum=FastSum(p=self.p,mac=self.mac,num_limit=self.num_limit,\
				 triangle_p=self.triangle_p,tetrahedron_p=self.tetrahedron_p)

        xt=self.mesh.coordinates()
        tet_nodes=np.array(self.mesh.cells(),dtype=np.int32)
        fast_sum.init_mesh(xt,self.t_normals,self.face_nodes_array,tet_nodes)
        self.fast_sum=fast_sum
	self.res=np.zeros(len(self.mesh.coordinates()))

	self.mesh.coordinates()[:]*=self.max_d
	


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


    def compute_field(self):
        
        m=self.m.vector().array()  
        
        self.fast_sum.update_charge(m)
        
        self.fast_sum.fastsum(self.res)
	
        self.fast_sum.compute_correction(m,self.res)
        
        self.phi.vector().set_local(self.res)
        
        self.phi.vector()[:]*=(self.Ms/(4*np.pi))
        
	demag_field = self.G * self.phi.vector()
        
	return demag_field.array()/self.L

if __name__ == "__main__":
   
    n=10
    #mesh = UnitCubeMesh(n, n, n)
    #mesh = Box(-1, 0, 0, 1, 1, 1, 10, 2, 2)
    mesh = UnitSphere(5)
    mesh.coordinates()[:]*=1
    
    Vv = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
    
    Ms = 8.6e5
    #expr = df.Expression(('cos(x[0])', 'sin(x[0])','0'))
    
    m = interpolate(Constant((1, 0, 0)), Vv)
    
    
    demag=Demag(triangle_p=1,tetrahedron_p=1,mac=0)
    demag.setup(Vv,m,Ms)
    demag.compute_field()
    print '='*100,'exact\n',demag.res
    exact=demag.res
    
    for p in [2,3,4,5,6,7]:
        demag=Demag(triangle_p=1,tetrahedron_p=1,mac=0.4,p=p)
        demag.setup(Vv,m,Ms)
        demag.compute_field()
        print '='*100,'mac=0.4 p=%d\n'%p,np.average(np.abs((demag.res-exact)/exact))
