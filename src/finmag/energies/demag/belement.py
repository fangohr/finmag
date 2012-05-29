import dolfin as df
import numpy as np
import instant
import belement_magpar
import finmag.util.solid_angle_magpar as solid_angle_solver
compute_belement=belement_magpar.return_bele_magpar()
compute_solid_angle=solid_angle_solver.return_csa_magpar()

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


def compute_bnd_mapping(mesh,debug=False):
    mesh.init()

    number_nodes=mesh.num_vertices()
    number_cells=mesh.num_cells()

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
        
    
    if debug:
        print 'cells number:',mesh.num_cells()
        print 'nodes number:',mesh.num_vertices() 
        #print mesh.coordinates()
        print 'faces:',mesh.num_faces()
        print 'faces number at the boundary:',number_faces_bnd
        print 'nodes number at the boundary:',number_nodes_bnd

        for i in range(number_nodes):
            tmp=gnodes_to_bnodes[i]
            print 'global id=',i,nodes_xyz[i][0],nodes_xyz[i][1],nodes_xyz[i][2],tmp
            
        for i in range(number_faces_bnd):
            print '   ',bnd_face_verts[i][0],bnd_face_verts[i][1],bnd_face_verts[i][2]
            
    
    return (bnd_face_verts,gnodes_to_bnodes,number_faces_bnd,number_nodes_bnd)


def BEM_matrix(mesh):
    bnd_face_verts,\
    gnodes_to_bnodes,\
    number_faces_bnd,\
    number_nodes_bnd=compute_bnd_mapping(mesh,debug=False)
    B=np.zeros((number_nodes_bnd,number_nodes_bnd))

    nodes_xyz=mesh.coordinates()
    tmp_bele=np.array([0.,0.,0.])

    number_nodes=mesh.num_vertices()

    for i in range(number_nodes):

        #skip the node at the boundary
        if gnodes_to_bnodes[i]<0:
            continue
        
        for j in range(number_faces_bnd):
            #skip the node in the face
            if i in set(bnd_face_verts[j]):
                continue

            compute_belement(nodes_xyz[i],
                        nodes_xyz[bnd_face_verts[j][0]],
                        nodes_xyz[bnd_face_verts[j][1]],
                        nodes_xyz[bnd_face_verts[j][2]],tmp_bele)
            """print 'tmp_bele',tmp_bele"""

            
            for k in range(3):
                tmp_i=gnodes_to_bnodes[i]
                tmp_j=gnodes_to_bnodes[bnd_face_verts[j][k]]
                B[tmp_i][tmp_j]+=tmp_bele[k]

    #the solid angle term ...
    vert_bsa=np.zeros(number_nodes)
    
    mapping_cell_nodes=mesh.cells()
    for i in range(mesh.num_cells()):
        for j in range(4):
            tmp_omega=compute_solid_angle(
                nodes_xyz[mapping_cell_nodes[i][j]],
                nodes_xyz[mapping_cell_nodes[i][(j+1)%4]],
                nodes_xyz[mapping_cell_nodes[i][(j+2)%4]],
                nodes_xyz[mapping_cell_nodes[i][(j+3)%4]])
            
            vert_bsa[mapping_cell_nodes[i][j]]+=tmp_omega

    for i in range(number_nodes):
        tmp_i=gnodes_to_bnodes[i]
        if tmp_i<0:
            continue
        
        B[tmp_i][tmp_i]+=vert_bsa[i]/(4*np.pi)-1

    return B
    

def test_order():
    mesh=df.Mesh('tet.xml')
    xs=mesh.coordinates()
    print xs
    print mesh.cells()

    print 'volume:',GetTetVol(xs[0],xs[1],xs[2],xs[3])
    print 'volume:',GetTetVol(xs[1],xs[0],xs[2],xs[3])

    print 'solid angle 1',compute_solid_angle(xs[0],xs[1],xs[2],xs[3])
    print 'solid angle 2',compute_solid_angle(xs[0],xs[2],xs[1],xs[3])


    bele=np.array([0,0.0,0])
    compute_belement(xs[0],xs[1],xs[2],xs[3],bele)
    print 'belement 1',bele
    
    compute_belement(xs[0],xs[2],xs[1],xs[3],bele)
    print 'belement 2',bele
    

if __name__=="__main__":

    #test_order()
    
    mesh=df.Mesh('cube.xml')
    print BEM_matrix(mesh)

