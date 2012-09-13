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
    
def compute_gauss_coeff_triangle(n):
    c=np.zeros(n*n)
    x=np.zeros(n*n)
    y=np.zeros(n*n)
    k=0
    xs=_nodes[n-1]
    ws=_weights[n-1]
    for i in range(n):
        for j in range(n):
            x[k]=(1+xs[i])/2.0
            y[k]=(1-xs[i])*(1+xs[j])/4.0
            c[k]=(1-xs[i])*ws[i]*ws[j]/8.0
            k+=1
    return x,y,c

def compute_gauss_coeff_tetrahedron(n):
    c=np.zeros(n**3)
    x=np.zeros(n**3)
    y=np.zeros(n**3)
    z=np.zeros(n**3)
    index=0
    xs=_nodes[n-1]
    ws=_weights[n-1]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                x[index]=(1+xs[i])/2.0
                y[index]=(1-xs[i])*(1+xs[j])/4.0
                z[index]=(1-xs[i])*(1-xs[j])*(1+xs[k])/8.0
                c[index]=(1-xs[i])*(1-xs[i])*(1-xs[j])*ws[i]*ws[j]*ws[k]/64.0
                index+=1
    return x,y,z,c


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
    
    def __init__(self,Vv, m, Ms,s_n=2,v_n=1):
        self.m=m
        self.Vv=Vv
        self.Ms=Ms
	self.s_n=s_n
        self.v_n=v_n
        self.mesh=Vv.mesh()
        self.V=FunctionSpace(self.mesh, 'Lagrange', 1)
        self.phi = Function(self.V)	

	self.node_vol=compute_node_volume(self.mesh)
	self.node_area=compute_node_area(self.mesh)
	self.cell_vol=compute_cell_volume(self.mesh)
	
	self.t_x,self.t_y,self.t_w=compute_gauss_coeff_triangle(s_n)
        self.v_x,self.v_y,self.v_z,self.v_w=compute_gauss_coeff_tetrahedron(v_n)
                
        u = TrialFunction(self.V)
        v = TestFunction(self.Vv)
        a = inner(grad(u), v)*dx
        self.G = df.assemble(a)
        
        self.L = compute_minus_node_volume_vector(self.mesh)
        
        self.compute_affine_transformation()
        self.compute_affine_transformation_volume()
        
        self.compute_charge_density()
        self.interpolate_at_surface()
        
        
        self.nodes=np.array(self.s_nodes+self.v_nodes)
        self.weights=np.array(self.s_weight+self.v_weight)
        
        self.charges=np.zeros(len(self.nodes))
        sn=len(self.s_weight)
        self.charges[:sn]=self.sigma_array[:]
        
        x_t=self.mesh.coordinates()
        x_s=self.nodes
        n=len(x_s)
        m=len(x_t)
        
        order=2*(int(n**(1.0/3)*0.64)+1)
        if order<64:
            order=64
        
        print 'n=%d  order=%d'%(n,order)
        
        fast_sum=FastSum(n=order)
        fast_sum.init_mesh(x_s,x_t)
        self.fast_sum=fast_sum
        
    
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
            
            for j in range(len(self.t_w)):
                x=c11*self.t_x[j]+c12*self.t_y[j]+x1
                y=c21*self.t_x[j]+c22*self.t_y[j]+y1
                z=c31*self.t_x[j]+c32*self.t_y[j]+z1
                
                self.s_nodes.append([x,y,z])
                self.s_weight.append(det*self.t_w[j])

        #self.s_nodes=np.array(self.s_nodes)
        #self.s_weight=np.array(self.s_weight)

    def compute_affine_transformation_volume(self):
        v = TestFunction(self.V)
        K = df.assemble(df.div(self.m) * v * df.dx)
        L = df.assemble(v * df.dx)
        rho = K.array()/L.array()
        print 'rho=\n',rho
        
        cs=self.mesh.coordinates()
        m=self.m.vector().array()
        n=len(m)/3
        print cs
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

            tmp=tmp/v
            return tmp,abs(v)
        
        self.v_nodes=[]
	self.v_weight=[]
        self.vol_density=[]
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

                self.vol_density.append(rho)
                self.v_nodes.append([x,y,z])
                self.v_weight.append(det*self.v_w[j])

        #self.v_nodes=np.array(self.v_nodes)
        #self.v_weight=np.array(self.v_weight)
        self.vol_density=np.array(self.vol_density)

    def compute_charge_density(self):

        m=self.m.vector().array()
        n=len(m)/3
        
        cf=np.zeros(len(self.face_nodes))
        for i in range(len(cf)):
            f_c=self.face_nodes[i]
    
            t=self.face_norms[i]
            for j in f_c:
		cf[i]+=(m[j]*t.x()+m[n+j]*t.y()+m[n*2+j]*t.z())*self.Ms/3.0
			
        self.sigma=cf
        

    def interpolate_at_surface(self):
        
        self.sigma_array=np.zeros(len(self.s_nodes))

        index=0
        for s in self.sigma:
            
            for i in range(len(self.t_w)):
                self.sigma_array[index]=s
                    
                index+=1
        

    def sum_using_nfft(self):

        m=len(self.mesh.coordinates())
        tmp_charge=self.weights*self.charges/(4*np.pi)
        self.fast_sum.update_charge(tmp_charge)
        res=np.zeros(m)
         
        self.fast_sum.compute_phi(res)    
        self.phi.vector().set_local(res)

    
    def compute_field(self):
	
	demag_field = self.G * self.phi.vector()
        
	return demag_field.array()/self.L
	    


    
if __name__ == "__main__":
   
    mesh = UnitSphere(5)
    mesh = UnitCube(2, 2, 2)
    
    mesh.coordinates()[:]*=0.25
    mesh.coordinates()[:]-=0.125
    
    Vv = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
    
    Ms = 8.6e5
    expr = df.Expression(('1+x[0]', '1+2*x[1]','1+3*x[2]'))
    m = project(expr, Vv)
    m = project(Constant((1, 0, 0)), Vv)
    

    demag=DemagNFFT(Vv,m,Ms,s_n=2)
    
    demag.sum_using_nfft()
    field=demag.compute_field()