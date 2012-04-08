from dolfin import *
import dolfin as df
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.dsolve import linsolve
import belement
import belement_magpar
import finmag.util.solid_angle_magpar as solid_angle_solver
compute_belement=belement_magpar.return_bele_magpar()
compute_solid_angle=solid_angle_solver.return_csa_magpar()


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


def compute_BEM_matrix(demag):
    """
    Input parameter is a SimpleFKSolver class
    """
    
    mesh=demag.mesh
    xyz=mesh.coordinates()
    bfn=demag.bnd_face_nodes
    g2b=demag.gnodes_to_bnodes

    nodes_number=mesh.num_vertices()
    
    n=demag.bnd_nodes_number
    B=np.zeros((n,n))

    tmp_bele=np.array([0.,0.,0.])
    
    for i in range(nodes_number):

        #skip the node not at the boundary
        if g2b[i]<0:
            continue
        
        for j in range(demag.bnd_faces_number):
            #skip the node in the face
            if i in set(bfn[j]):
                continue

            compute_belement(
                xyz[i],
                xyz[bfn[j][0]],
                xyz[bfn[j][1]],
                xyz[bfn[j][2]],
                tmp_bele)
           
            
            for k in range(3):
                ti=g2b[i]
                tj=g2b[bfn[j][k]]
                B[ti][tj]+=tmp_bele[k]

    #the solid angle term ...
    vert_bsa=np.zeros(nodes_number)
    

    mc=mesh.cells()
    for i in range(mesh.num_cells()):
        for j in range(4):
            tmp_omega=compute_solid_angle(
                xyz[mc[i][j]],
                xyz[mc[i][(j+1)%4]],
                xyz[mc[i][(j+2)%4]],
                xyz[mc[i][(j+3)%4]])
            
            vert_bsa[mc[i][j]]+=tmp_omega

    for i in range(nodes_number):
        j=g2b[i]
        if j<0:
            continue
        
        B[j][j]+=vert_bsa[i]/(4*np.pi)-1

    demag.B=B


"""
just intend to demostrate how to compute the demag field using FK method
so the programming style and efficiency are not considered now ...
"""
class SimpleFKSolver():
    
    def __init__(self,Vv, m, Ms):
        self.m=m
        self.Vv=Vv
        self.Ms=Ms
        self.mesh=Vv.mesh()
        self.V=FunctionSpace(self.mesh, 'Lagrange', 1)
        self.phi1 = Function(self.V)
        self.phi2 = Function(self.V)

        self.__bulid_Mapping()
        self.__bulid_Matrix(debug=False)
        
        
    def __bulid_Mapping(self):
        self.bnd_face_nodes,\
        self.gnodes_to_bnodes,\
        self.bnd_faces_number,\
        self.bnd_nodes_number= \
                belement.compute_bnd_mapping(self.mesh)

        self.nodes_number=self.mesh.num_vertices()

    def __bulid_Matrix(self,debug=False):
        
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        n = FacetNormal(self.mesh)

        #=============================================
        """
        K1 * phi1 = g1
        """
        a = inner(grad(u),grad(v))*dx 
        self.K1=df.assemble(a)

        if debug:
            print '='*100,'K1\n',self.K1.array()

        f = self.Ms*(dot(n, self.m)*v*ds - div(self.m)*v*dx)
        self.g1=df.assemble(f)
        
        if debug:
            print '='*100,'g1\n',self.g1.array()
        
        #=============================================
        """
        U1 * phi1 = u1bnd
        """
        self.U1= sp.lil_matrix((self.bnd_nodes_number,
                                   self.nodes_number),
            dtype='float32')

        g2b=self.gnodes_to_bnodes
        for i in range(self.nodes_number):
            if g2b[i]>=0:
                self.U1[g2b[i],i]=1

        if debug:
            print '='*100,'U1\n',self.U1
        
        #=============================================
        """
        B * u1bnd = u2bnd
        """
        
        compute_BEM_matrix(self)
        
        if debug:
            print '='*100,'B\n',self.B

        #=============================================
        """
        U2 * u2bnd = g2
        """
        #in fact, I am not sure whether the following method is correct ...
        self.U2= sp.lil_matrix((self.nodes_number,
                                self.bnd_nodes_number),
                            dtype='float32')

        g2b=self.gnodes_to_bnodes

        tmp_mat=sp.lil_matrix(self.K1.array())
        rows,cols = tmp_mat.nonzero()
        
        for row,col in zip(rows,cols):
            if g2b[row]<0 and g2b[col]>=0:
                self.U2[row,g2b[col]]=-tmp_mat[row,col]
                
        for i in range(self.nodes_number):
            if g2b[i]>=0:
                self.U2[i,g2b[i]]=1
                
        if debug:
            print '='*100,'U2\n',self.U2

        #=============================================
        """
        K2 * phi2 = g2
        """
        self.K2= sp.lil_matrix((self.nodes_number,
                                self.nodes_number),
                            dtype='float32')
        
        tmp_mat=sp.lil_matrix(self.K1.array())
        rows,cols = tmp_mat.nonzero()
        
        for row,col in zip(rows,cols):
            if g2b[row]<0 and g2b[col]<0:
                self.K2[row,col]=tmp_mat[row,col]
                
        for i in range(self.nodes_number):
            if g2b[i]>=0:
                self.K2[i,i]=1    

        """
        self.K2=df.assemble(a)
        
        def zero_boundary(x, on_boundary):
            return on_boundary
        
        bc = df.DirichletBC(self.V, 0, zero_boundary)
        bc.apply(self.K2)
        """


        if debug:
            print '='*100,'K2\n',self.K2

        #=============================================
        """
        I know we can use "project" function to obtain the magnetisation from the total phi
        and it should be correct, Yes, but it seems that one can use some matrix method such
        as in the reference

           "Numerical Methods in Micromagnetics (Finite Element Method)
            Thomas Schrefl1, Gino Hrkac1, Simon Bance1, Dieter Suess2, Otmar Ertl2 and Josef Fidler"

        the magnetisation was extracted by multiplying some matrix,

              H_demag= - L^-1 * G * phi

        Is this method the same with the project function? According to the instrcutions,

              demag_field = df.project(-df.grad(phi), self.Vv)

        is equivalent to the follwoing statements:

              w = TrialFunction(self.Vv)
              v = TestFunction(self.Vv)

              a = inner(w, v)*dx
              L = inner(grad(u), v)*dx
              demag_field = Function(self.Vv)
              solve(a == L, demag_field)

        Another question is how assemble the above statements into matrix form? I am lost in
        dolfin now, ;-), I only arrived at

              u=TrialFunction(self.V)
              w = TrialFunction(self.Vv)
              v = TestFunction(self.Vv)

              a = inner(w, v)*df.dx
              A=df.assemble(a)

              b = inner(grad(u), v)*df.dx
              B=df.assemble(b)

              demag_field = Function(self.Vv)
              solve(A, demag_field.vector(),B*phi)

        but accoring to numerical experiments, I found B was the matrix we are looking for
        and L is certainly the volume of each node. I was surprised becasue this means the
        the following eqaution holds,

              - L^-1 B = A B

        I have no idea wether it is always true?  why?
        
        """

        u = TrialFunction(self.V)
        v = TestFunction(self.Vv)
        a= inner(grad(u), v)*dx
        self.G = df.assemble(a)
        
        self.L = compute_minus_node_volume_vector(self.mesh)


    def compute_field(self,debug=False):
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        n = FacetNormal(self.mesh)

        
        solve(self.K1,self.phi1.vector(),self.g1)
        if debug:
            print '='*100,'phi1\n',self.phi1.vector().array()

        self.u1bnd=self.U1*self.phi1.vector()
        if debug:
            print '='*100,'u1bnd ',type(self.u1bnd),self.u1bnd.shape,'\n',self.u1bnd

        self.u2bnd=np.dot(self.B,self.u1bnd)
        if debug:
            print '='*100,'u2bnd\n',self.u2bnd

        self.g2=self.U2*self.u2bnd
        if debug:
            print '='*100,'g2\n',self.g2

        self.K2=self.K2.tocsr()
        phi2=linsolve.spsolve(self.K2,self.g2,use_umfpack = False)
        if debug:
            print '='*100,'phi2\n',phi2
    
        
        phi=Function(self.V)
        phi.vector().set_local(self.phi1.vector().array()+phi2)
        #demag_field = df.project(-df.grad(phi), self.Vv)
        
        demag_field= self.G * phi.vector()
        
        return demag_field.array()/self.L


if __name__ == "__main__":

    from finmag.demag.problems.prob_fembem_testcases import MagSphere
    mesh = MagSphere(5,1.).mesh
    print mesh

    V = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
    Ms=8.6e5

    m = project(Constant((1, 0, 0)), V)
    demag=SimpleFKSolver(V,m,Ms)
    print demag.compute_field()
    
    
