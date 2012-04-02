import numpy as np
import dolfin as df
from dolfin import *
import numpy as np
import os
import subprocess

def set_inital_m0(V,m0):
    if isinstance(m0, tuple):
            if isinstance(m0[0], str):
                val = df.Expression(m0)
            else:
                val = df.Constant(m0)
            m = df.interpolate(val, V)
            return m
    else:
        raise NotImplementedError,"only a tuple is acceptable for set_inital_m0"

def gen_magpar_conf(base_name,init_m,Ms=8.6e5,A=13e-12,K1=0,a=[0,0,1],alpha=0.1,demag=0):
    save_path=os.getcwd()
    new_path=os.path.join(save_path,base_name)
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    norm_a=(a[0]**2+a[1]**2+a[2]**2)**0.5
    tmp_mz=a[2]/norm_a

    theta=np.arccos(tmp_mz)
    if a[0]==0:
        phi=0
    else:
        phi=np.arctan(a[1]/a[0])
    
    # theta phi   K1       K2      Js   A        alpha psi   # parameter
    # (rad) (rad) (J/m^3)  (J/m^3) (T)  (J/m)    (1)   (rad) # units
    krn_info="  %f   %f   %e    0     %f    %e   %f   uni"%(
        theta,phi,K1,np.pi*4e-7*Ms,A,alpha
        )
   
    allopt=["-simName ",base_name+"\n",
            "-init_mag ","0\n",
            "-inp ","0000\n",
            "-demag ", str(demag) ]

    file_name=os.path.join(new_path,base_name+".krn")
    f=open(file_name,'w')
    f.write(krn_info)
    f.close()

    file_name=os.path.join(new_path,"allopt.txt")
    f=open(file_name,'w')
    f.write("".join(allopt))
    f.close()
    
    file_name=os.path.join(new_path,base_name+".inp")
    save_inp_of_inital_m(init_m,file_name)

    file_name=os.path.join(new_path,base_name+".0000.inp")
    save_inp_of_inital_m(init_m,file_name)
    
    

def run_magpar(base_name):
    magpar_cmd=(os.path.join(os.getenv("HOME")+"/magpar-0.9/src/magpar.exe"))
    
    save_path=os.getcwd()
    new_path=os.path.join(save_path,base_name)
    os.chdir(new_path)
    print new_path
    subprocess.check_call(magpar_cmd,stdout=subprocess.PIPE)

    gzcmd=("gunzip",base_name+".0001.gz")
    subprocess.check_call(gzcmd)
    os.chdir(save_path)

def read_femsh(file_name):
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

    return np.array(node_coord),np.array(connectivity)
    

def read_inp_gz(file_name):
    f=open(file_name,'r')
    a=f.readline()
    num=int(a.split()[0])
    names=[]
    for i in range(num):
        names.append(f.readline().split(',')[0])
    
    lines=f.readlines()
    f.close()

    data=[]
    for line in lines:
        data.append([float(t) for t in line.split()])


    data=np.array(data)
    
    fields={}
    for i in range(num):
        fields[names[i]]=data[:,i+1]
    
    return fields
    

def save_inp_of_inital_m(m,file_name):
    mesh=m.function_space().mesh()
    data_type_number = 3
    f=open(file_name,"w")
    head="%d %d %d 0 0\n" % (
        mesh.num_vertices(),
        mesh.num_cells(),
        data_type_number)
    f.write(head)
    xyz=mesh.coordinates()
    for i in range(len(xyz)):
        f.write("%d %f %f %f\n"
                %(i+1,
                  xyz[i][0]*1e9,
                  xyz[i][1]*1e9,
                  xyz[i][2]*1e9))
    
    for c in cells(mesh):
        id=c.index()
        ce=c.entities(0)
        f.write("%d 1 tet %d %d %d %d\n"
                %(id+1,
                  ce[0]+1,
                  ce[1]+1,
                  ce[2]+1,
                  ce[3]+1))
    f.write("3 1 1 1\nM_x, none\nM_y, none\nM_z, none\n")

    data=m.vector().array().reshape(3,-1)
    for i in range(mesh.num_vertices()):
        f.write("%d %e %e %e\n"
                %(i+1,
                  data[0][i],
                  data[1][i],
                  data[2][i]))

    f.close()
    

def get_field(base_name,field="anis"):
    new_path=os.path.join(os.getcwd(),base_name)
    file_name=os.path.join(new_path,base_name+".0001")
    fields=read_inp_gz(file_name)
    
    if field=="anis":
        fx=fields["Hani_x"]
        fy=fields["Hani_y"]
        fz=fields["Hani_z"]
    elif field=="exch":
        fx=fields["Hexch_x"]
        fy=fields["Hexch_y"]
        fz=fields["Hexch_z"]
    else:
        raise NotImplementedError,"only exch and anis field can be extracted now"

    field=np.array([fx,fy,fz]).reshape(1,-1,order='F')[0]
    field=field/(np.pi*4e-7)

    file_name=os.path.join(new_path,base_name+".0001.femsh")
    nodes,connectivity=read_femsh(file_name)

    return nodes,field


def compute_anis_magpar(V, m, K, a, Ms):
    """
    Usage:

    mesh = Box(0, 10e-9, 0, 1e-9, 0, 1e-9, 5, 1, 1)

    V = VectorFunctionSpace(mesh, 'Lagrange', 1)
    K = 520e3 # For Co (J/m3)

    a = [0,0,1] # Easy axis in z-direction
    
    m0_x = "pow(sin(x[0]*1e9), 2)"
    m0_y = "0"
    m0_z = "pow(cos(x[0]*1e9), 2)"
    m=set_inital_m0(V,(m0_x,m0_y, m0_z))
    Ms=14e5 

    anisotropy = compute_anis_magpar(V, m, K, a, Ms)
    """
    base_name="test_anis"

    gen_magpar_conf(base_name,m,Ms=Ms,a=a,K1=K)
  
    run_magpar(base_name)
 
    nodes,field=get_field(base_name,field="anis")
      
    new_path=os.path.join(os.getcwd(),base_name)
    rm_cmd=("rm","-rf",new_path)
    subprocess.check_call(rm_cmd)
    
    return nodes,field


def get_m0(file_name):
    fields=read_inp_gz(file_name)
    fx=fields["M_x"]
    fy=fields["M_y"]
    fz=fields["M_z"]
    
    field=np.array([fx,fy,fz]).reshape(1,-1)[0]
    return field
    
def compute_exch_magpar(V, m, C, Ms):
    """
    Usage:

    mesh = Box(0, 10e-9, 0, 1e-9, 0, 1e-9, 5, 1, 1)

    V = VectorFunctionSpace(mesh, 'Lagrange', 1)
    K = 520e3 # For Co (J/m3)

    a = [0,0,1] # Easy axis in z-direction
    
    m0_x = "pow(sin(x[0]*1e9), 2)"
    m0_y = "0"
    m0_z = "pow(cos(x[0]*1e9), 2)"
    m=set_inital_m0(V,(m0_x,m0_y, m0_z))
    Ms=14e5 
    C=13e-12

    anisotropy = compute_exch_magpar(V, m, C, Ms)
    """
    base_name="test_exch"


    gen_magpar_conf(base_name,m,Ms=Ms,A=C)
    
    new_path=os.path.join(os.getcwd(),base_name)
    run_magpar(base_name)

    nodes,field=get_field(base_name,field="exch")
      
    new_path=os.path.join(os.getcwd(),base_name)

    #remove files from temporary directory
    rm_cmd=("rm","-rf",new_path)
    subprocess.check_call(rm_cmd)

    return nodes,field

    
    
if __name__=="__main__":
        
    mesh = Box(0, 10e-9, 0, 1e-9, 0, 1e-9, 5, 1, 1)

    V = VectorFunctionSpace(mesh, 'Lagrange', 1)

    K = 520e3 # For Co (J/m3)
    a = [2,0,1] # Easy axis in z-direction
    Ms = 14e5 

    m=set_inital_m0(V,(0,0.6,0.8))
    nodes,anis = compute_anis_magpar(V, m, K, a, Ms)
    print nodes,anis
    
    C=13e-12
    nodes,exch = compute_exch_magpar(V, m, C, Ms)
    print exch
    
    

