from dolfin import *
import dolfin as df
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from demag_nfft_lib import FastSum 


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
    
def compute_gauss_coeff(n):
    c=np.zeros(n*n)
    x=np.zeros(n*n)
    y=np.zeros(n*n)
    k=0
    xs=_nodes[n-1]
    ws=_weights[n-1]
    for i in range(n):
        for j in range(n):
            c[k]=(1-xs[i])*ws[i]*ws[j]/8.0
            x[k]=(1+xs[i])/2.0
            y[k]=(1-xs[i])*(1+xs[j])/4.0
            k+=1
    return x,y,c


def length(p1,p2):
	return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)

def length2(p1,p2):
	return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2

def G(r1,r2):
	r=length(r1,r2)
	
	if r<1e-12:
            print 'hhhhh'
            return 0
		
	return 1/(4*np.pi*r)

def mapping_3d_triangle_2d(p1,p2,p3):
    """
    mapping a 3d triangle to xy plane with the same shape and size
    p1-->(0,0)
    p2-->(w,0)
    p3-->(u,v)
    """
    w=length(p1,p2)
    a=length2(p1,p3)
    b=length2(p2,p3)
    u=(a-b)/(2*w)+w/2.0
    v=np.sqrt(2*w**2*(a+b)-(a-b)**2-w**4)/(2*w)
    return w,u,v

def compute_correction_coeff(w,u,v):
    """
    We knew the charge density at three nodes of a triangle, say sigma1,sigma2,sigma3
    then the exact solution for 1/r integration over the triangle with assumption that
    the charge density changing linearly is:
    
       (sigma2-simga1)*a+(sigma3-simga1)*b+sigma1*c

    where
        a,b,c are computed by this function.
        
    """
    det=w*v
    q=np.sqrt(u*u + v*v)
    p=np.sqrt(u*u + v*v - 2*u*w + w*w)
    tmp=np.log((p+u-w)*w)-np.log(p*q + q*q - u*w)

    a=(p*(w-q) + (u*w-q*q)*tmp)/(2*p**3)*det
    b=(p*(q-w) + (u-w)*w*tmp)/(2*p**3)*det
    c=-tmp/p*det
    
    return a,b,c


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

def plot_points(data1,data2):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for c, m, data in [('r', 'o', data1), ('b', '^', data2)]:
        tmp=np.array(data)
        xs = tmp[:,0]
        ys = tmp[:,1]
        zs = tmp[:,2]
        ax.scatter(xs, ys, zs, c=c, marker=m)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


class DemagNFFT():
    
    def __init__(self,Vv, m, Ms,t_n=3):
        self.m=m
        self.Vv=Vv
        self.Ms=Ms
	self.t_n=t_n
        self.mesh=Vv.mesh()
        self.V=FunctionSpace(self.mesh, 'Lagrange', 1)
        self.phi = Function(self.V)
	

	self.node_vol=compute_node_volume(self.mesh)
	self.node_area=compute_node_area(self.mesh)
	self.cell_vol=compute_cell_volume(self.mesh)
	
	self.t_x,self.t_y,self.t_w=compute_gauss_coeff(t_n)
        """
        print self.t_x
        print self.t_y
        print self.t_w
        """
        u = TrialFunction(self.V)
        v = TestFunction(self.Vv)
        a = inner(grad(u), v)*dx
        self.G = df.assemble(a)
        
        self.L = compute_minus_node_volume_vector(self.mesh)
        
    
    
    def compute_affine_transformation(self):
	"""
	b_mesh=df.BoundaryMesh(self.mesh)
	print b_mesh
	for face in df.faces(b_mesh):
		face_nodes=face.entities(0)
		print face_nodes
	"""
    

	cs=self.mesh.coordinates()
        self.face_nodes=[]
        self.face_norms=[]
	for face in df.faces(self.mesh):
		cells = face.entities(3)
		if len(cells)==1:
			face_nodes=face.entities(0)
                        self.face_nodes.append(face_nodes)
                        self.face_norms.append(face.normal())
			

        self.s_nodes=[]
	self.s_weight=[]

        def compute_det_xy(x1,y1,z1,x2,y2,z2,x3,y3,z3):
           
            a = y2*z1 - y3*z1 - y1*z2 + y3*z2 + y1*z3 - y2*z3
            b = x2*z1 - x3*z1 - x1*z2 + x3*z2 + x1*z3 - x2*z3
            c = x2*y1 - x3*y1 - x1*y2 + x3*y2 + x1*y3 - x2*y3

            det=abs((x2-x1)*(y3-y1)-(y3-y1)*(y2-y1))

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
            
            for j in range(len(self.t_w)):
                x=c11*self.t_x[j]+c12*self.t_y[j]+x1
                y=c21*self.t_x[j]+c22*self.t_y[j]+y1
                z=c31*self.t_x[j]+c32*self.t_y[j]+z1
                
                self.s_nodes.append([x,y,z])
                self.s_weight.append(det*self.t_w[j])

        self.s_nodes=np.array(self.s_nodes)
        self.s_weight=np.array(self.s_weight)
                

    def compute_charge_density(self):

        m=self.m.vector().array()
        n=len(m)/3
        
        cf=np.zeros(len(self.face_nodes))
        for i in range(len(cf)):
            f_c=self.face_nodes[i]
    
            t=self.face_norms[i]
            for j in f_c:
		cf[i]+=(m[j]*t.x()+m[n+j]*t.y()+m[n+2*j]*t.z())*self.Ms/3.0
			
        self.sigma=cf
        

    def interpolate_at_surface(self):
        #plot_points(self.s_nodes,self.mesh.coordinates())
        
        self.sigma_array=np.zeros(len(self.s_nodes))

        index=0
        for s in self.sigma:
            
            for i in range(len(self.t_w)):
                self.sigma_array[index]=s
                    
                index+=1
        
    
    def sum_directly(self):
        cs=self.mesh.coordinates()
        m=len(cs)
        n=len(self.s_nodes)
        print 'n=',n
        
        self.interpolate_at_surface()
        
        res=np.zeros(m)
        for i in range(m):
            for j in range(n):
                res[i]+=G(cs[i],self.s_nodes[j])*self.s_weight[j]*self.sigma_array[j]

        self.phi.vector().set_local(res)
        #print res

    def sum_using_nfft(self):
        x_t=self.mesh.coordinates()
        x_s=self.s_nodes
        n=len(x_s)
        m=len(x_t)
        print 'm,n=',m,n
        
        self.interpolate_at_surface()
        

        fast_sum=FastSum()
        fast_sum.init_mesh(x_s,x_t)
        tmp_charge=self.s_weight*self.sigma_array/(4*np.pi)
        fast_sum.update_charge(tmp_charge)
        res=np.zeros(m)
        
        fast_sum.sum_exact(res)
        print 'sum directly:\n',res
        
        fast_sum.compute_phi(res)
    
        self.phi.vector().set_local(res)

    def compute_correction(self):
        cs=self.mesh.coordinates()
        m=len(cs)
        self.correction_phi=np.zeros(m)
        
        
        def correct_over_triangle(self,base_index,i1,i2,i3,sigma,cs=cs):
            
            p1=cs[i1]
            p2=cs[i2]
            p3=cs[i3]
            
            guass_sum=0
            for i in range(len(self.t_w)):
                j=base_index+i
                guass_sum+=G(p1,self.s_nodes[j])*self.sigma_array[j]*self.s_weight[j]
                

            w,u,v=mapping_3d_triangle_2d(p1,p2,p3)
            a,b,c=compute_correction_coeff(w,u,v)
            exact=c*sigma/(4*np.pi)
            
            """
            print a,b,c,u1,u2,u3
            print i1,i2,i3
            """
            #print exact,guass_sum,abs((guass_sum-exact)/exact)
            
            self.correction_phi[i1]+=exact-guass_sum
            
        
        base_index=0
        index=0
        for f_n in self.face_nodes:
            correct_over_triangle(self,base_index,f_n[0],f_n[1],f_n[2],self.sigma[index])
            correct_over_triangle(self,base_index,f_n[1],f_n[2],f_n[0],self.sigma[index])
            correct_over_triangle(self,base_index,f_n[2],f_n[0],f_n[1],self.sigma[index])
            index+=1
            base_index+=len(self.t_w)

        self.phi.vector()[:]+=self.correction_phi
            

    def compute_field(self):
	
	demag_field = self.G * self.phi.vector()
        
	return demag_field.array()/self.L
	    

    
if __name__ == "__main__":
   
    mesh = UnitSphere(5)
    mesh = UnitCube(4, 4, 4)
    
    mesh.coordinates()[:]*=0.2
    
    Vv = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
    
    Ms=8.6e5
    m = project(Constant((1, 0, 0)), Vv)
    
      
    def compute_error(field):
        
        field.shape=((3,-1))
    
        fem=271330.842297
        
        hx=np.average(field[0])
        return hx,(fem+hx)/fem
    
    for tn in range(1,2):
        demag=DemagNFFT(Vv,m,Ms,t_n=tn)
        demag.compute_affine_transformation()
        demag.compute_charge_density()
        
        demag.sum_using_nfft()
        print 'sum_nfft\n',demag.phi.vector().array()
        """
        field=demag.compute_field()
        print 't_n=',tn,compute_error(field),'after correction',
        
        demag.compute_correction()
        print demag.phi.vector().array()
        field=demag.compute_field()
        print compute_error(field)
        """
        #print field
    
    

    

