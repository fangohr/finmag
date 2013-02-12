##import io
import numpy as np
import dolfin as df
import os
import logging
import subprocess

logger = logging.getLogger(name='finmag')
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def gen_magpar_conf(base_name, init_m,
        Ms=8.6e5, A=13e-12, K1=0, a=[0,0,1], alpha=0.1, demag=0):

    conf_path = os.path.join(MODULE_DIR, base_name)
    if not os.path.exists(conf_path):
        os.makedirs(conf_path)
    logger.debug("Saving magpar files to {}.".format(conf_path))

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
    with open(os.path.join(conf_path, base_name+".krn"), "w") as krn_file:
        krn_file.write(krn_info)

    allopt=["-simName ",base_name+"\n",
            "-init_mag ","0\n",
            "-inp ","0000\n",
            "-demag ", str(demag) ]
    with open(os.path.join(conf_path, "allopt.txt"), "w") as allopt_file:
        allopt_file.write("".join(allopt))
   
    file_name=os.path.join(conf_path, base_name+".inp")
    save_inp_of_inital_m(init_m,file_name)

    file_name=os.path.join(conf_path, base_name+".0000.inp")
    save_inp_of_inital_m(init_m,file_name)

def run_magpar(base_name):
    magpar_cmd=(os.path.join("magpar.exe"))
   
    save_path=os.getcwd()
    new_path=os.path.join(MODULE_DIR, base_name)
    os.chdir(new_path)

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
        line = f.readline()
        tmp=line.split()
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
    if np.max(xyz) < 0.5:
       print "Converting unit_length from m to nm."
       xyz = xyz * 1e9
    for i in range(len(xyz)):
        f.write("%d %0.15e %0.15e %0.15e\n"
                %(i+1,
                  xyz[i][0],
                  xyz[i][1],
                  xyz[i][2]))
    
    for c in df.cells(mesh):
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
        f.write("%d %0.15e %0.15e %0.15e\n"
                %(i+1,
                  data[0][i],
                  data[1][i],
                  data[2][i]))

    f.close()
    

def get_field(base_name,field="anis"):
    new_path=os.path.join(MODULE_DIR,base_name)
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
    elif field=="demag":
        fx=fields["Hdemag_x"]
        fy=fields["Hdemag_y"]
        fz=fields["Hdemag_z"]
    else:
        raise NotImplementedError,"only exch and anis field can be extracted now"

    field=np.array([fx,fy,fz]).reshape(1,-1,order='C')[0]
    field=field/(np.pi*4e-7)

    file_name=os.path.join(new_path,base_name+".0001.femsh")
    nodes,connectivity=read_femsh(file_name)

    return nodes,field

def get_m0(file_name):
    fields=read_inp_gz(file_name)
    fx=fields["M_x"]
    fy=fields["M_y"]
    fz=fields["M_z"]
    
    field=np.array([fx,fy,fz]).reshape(1,-1)[0]
    return field
 
def compute_anis_magpar(m, **kwargs):
    return compute("anis", m, **kwargs)
   
def compute_exch_magpar(m, **kwargs):
    return compute("exch", m, **kwargs)

def compute_demag_magpar(m, **kwargs):
    return compute("demag", m, demag=1, **kwargs)

def compute(field_name, m, **kwargs):
    base_name = "test_" + field_name

    gen_magpar_conf(base_name, m, **kwargs)
    run_magpar(base_name)

    nodes, field = get_field(base_name, field_name)

    delete = ("rm", "-rf", os.path.join(MODULE_DIR, base_name))
    subprocess.check_call(delete)

    return nodes, field
 
def compare_field(aNodes, aField, bNodes, bField):
    """
    Compares two vector fields aField and bField defined over the meshes
    aNodes and bNodes respectively.

    When n is the number of nodes, we expect aField and bField to be
    ndarrays of shape 3n, and aNodes and bNodes to be ndarrays of shape (n, 3).

    """
    assert aNodes.shape == bNodes.shape
    assert aField.shape == bField.shape

    aField = aField.reshape((3, -1))
    bField = bField.reshape((3, -1))

    bFieldOrdered = np.zeros(bField.shape)
    for i, aNode in enumerate(aNodes):
        closest_bNode_index = np.abs(bNodes - aNode).mean(axis=1).argmin()
        for dim in range(3):
            bFieldOrdered[dim][i] = bField[dim][closest_bNode_index]

    diff = np.abs(bFieldOrdered - aField)
    rel_diff = diff / np.sqrt(np.max(bFieldOrdered[0]**2 + bFieldOrdered[1]**2 + bFieldOrdered[2]**2))
  
    return aField, bFieldOrdered, diff, rel_diff

def compare_field_directly(node1,field1,node2,field2):
    """
    Compares field1 defined over the nodes on node1 to field2 over the nodes
    node2. Note that the nodes should be the same, but are allowed to be
    in different order.

    acceptable fields should like this:
    [fx0, ..., fxn, fy0, ..., fyn, fz0, ..., fzn]
    """
    assert node1.shape == node2.shape
    assert field1.shape == field2.shape

    field1=field1.reshape(3,-1)
    field2=field2.reshape(3,-1)

    # build two dicts of coordinates -> values in order
    # to sort the data of field2 by the ordering of the data in field1.
    key1=[]
    key2=[]
    data2={}
    for i in range(len(node1)):
        tmp1="%f%f%f"%(node1[i][0],node1[i][1],node1[i][2])
        tmp2="%f%f%f"%(node2[i][0],node2[i][1],node2[i][2])
        key1.append(tmp1)
        key2.append(tmp2)
        data2[tmp2]=[field2[0][i],field2[1][i],field2[2][i]]

    assert(set(key1)==set(key2))

    field2_ordered=np.array([data2[k] for k in key1])
    field2_ordered= field2_ordered.reshape(1,-1)[0]
    field2_ordered= field2_ordered.reshape(3,-1,order='F')
   
    difference = np.abs(field2_ordered - field1)
    relative_difference = difference / np.max(np.sqrt(
        field2_ordered[0]**2 + field2_ordered[1]**2 + field2_ordered[2]**2))

    return field1,field2_ordered,difference,relative_difference
