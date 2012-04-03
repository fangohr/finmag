import os
import dolfin as df
import numpy as np
import magpar
from finmag.sim.anisotropy import UniaxialAnisotropy

from finmag.sim.helpers import quiver, boxplot, stats
#df.parameters["allow_extrapolation"] = True


REL_TOLERANCE = 2e-1 # goal: < 1e-3
MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"


def read_inp(file_name):
    f=open(file_name,'r')
    a=f.readline()
    n_node=int(a.split()[0])
    n_cell=int(a.split()[1])

    node_coord=[]
    for i in range(n_node):
        tmp=f.readline().split()
        t2=[float(tmp[1]),float(tmp[2]),float(tmp[3])]
        node_coord.append(t2)
    
    connectivity=[]
    for i in range(n_cell):
        t=f.readline().split()
        t2=[int(t[3]),int(t[4]),int(t[5]),int(t[6])]
        connectivity.append(t2)

    a=f.readline()
    num=int(a.split()[0])
    names=[]
    for i in range(num):
        names.append(f.readline().split(',')[0])
    
    lines=f.readlines()
    f.close()

    data=[]
    for line in lines:
        data.append([float(t)/(np.pi*4e-7) for t in line.split()])


    data=np.array(data)
    
    fields={}
    fields['nodes']=np.array(node_coord)
    for i in range(num):
        fields[names[i]]=data[:,1+3*i:4+3*i].reshape(1,-1,order='F')[0]
    
    return fields



def three_dimensional_problem():
    x_max = 10e-9; y_max = 1e-9; z_max = 1e-9;
    mesh = df.Box(0, 0, 0, x_max, y_max, z_max, 40, 2, 2)
    
    V = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
    K = 520e3 # For Co (J/m3)
    Ms=45e4

    a = (0,0,1) # Easy axis in z-direction

    m0_x = "pow(sin(0.2*x[0]*1e9), 2)"
    m0_y = "0"
    m0_z = "pow(cos(0.2*x[0]*1e9), 2)"
    m=magpar.set_inital_m0(V,(m0_x,m0_y, m0_z))

    u_anis = UniaxialAnisotropy(V, m, K, df.Constant(a), Ms)
    finmag_anis = u_anis.compute_field()
 
    mumag_data=read_inp("test_anis-my_code.inp")
    mumag_anis=mumag_data['Hanis']

    nodes=mumag_data['nodes'] 
          
    tmp=df.Function(V)
    tmp_c = mesh.coordinates()
    mesh.coordinates()[:]=tmp_c*1e9
    
    finmag_anis,mumag_anis, \
        diff,rel_diff=magpar.compare_field_directly( \
            mesh.coordinates(),finmag_anis,\
            nodes, mumag_anis)

    return dict( m0=m.vector().array(),
                 mesh=mesh,
                 anis=finmag_anis,
                 mumag_anis=mumag_anis,
                 diff=diff, 
                 rel_diff=rel_diff)


if __name__ == '__main__':

    
    res = three_dimensional_problem()
   
    print "finmag:",res["anis"]
    print "mumag:",res["mumag_anis"]
    print "rel_diff:",res["rel_diff"]
    print "max rel_diff",np.max(res["rel_diff"])

   

