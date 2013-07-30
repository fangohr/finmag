import dolfin as df
import numpy as np
import logging
from finmag.util.consts import mu0
from finmag.util.timings import mtimed
from finmag.util import helpers
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.sparse.linalg.dsolve import linsolve

logger=logging.getLogger('finmag')

"""
Compute the exchange field in DG0 space with the help of BDM1 space.

With the known magnetisation m in DG space, its gradient sigma in BDM 
space can be obtained by solving the linear equation:

    A sigma = K1 m

then the exchange fields F can be approached by 

    F = K2 sigma
    
""" 
def copy_petsc_to_csc(pm):
    (m,n) = pm.size(0), pm.size(1)
    matrix = sp.lil_matrix((m,n))
    for i in range(m):
        ids, values = pm.getrow(i)
        matrix[i,ids] = values

    return matrix.tocsc()

def copy_petsc_to_csr(pm):
    (m,n) = pm.size(0), pm.size(1)
    matrix = sp.lil_matrix((m,n))
    for i in range(m):
        ids, values = pm.getrow(i)
        matrix[i,ids] = values
    
    return matrix.tocsr()



def sparse_inverse(A):
    """
    suppose A is a sparse matrix and its inverse also a sparse matrix.
    seems it's already speedup a bit, but we should be able to do better later.
    """
    solve = spl.factorized(A)
    n = A.shape[0]
    assert (n == A.shape[1])
    mat = sp.lil_matrix((n,n))
    
    for i in range(n):
        b = np.zeros(n)
        b[i] = 1
        x = solve(b)
        ids =  np.nonzero(x)[0]
        
        for id in ids:
            mat[id,i] = x[id]

    return mat.tocsc()

def generate_nonzero_ids(mat):
    """
    generate the nonzero column ids for every rows
    """
       
    idxs,idys = mat.nonzero()
    
    max_x=0
    for x in idxs:
        if x>max_x:
            max_x=x
    idy=[]
    for i in range(max_x+1):
        idy.append([])

    for i, x in enumerate(idxs):
        idy[x].append(idys[i])
     
    assert(len(idy)==max_x+1)
    return np.array(idy)


def compute_nodal_triangle():
    """
    The nodal vectors are computed using the following normals
        n0 = np.array([1,1])/np.sqrt(2)
        n1 = np.array([1.0,0])
        n2 = np.array([0,-1.0])
    """
    
    v0=[[0,0],[0,0],[1,0],[0,0],[0,-1],[0,0]]
    v1=[[1,0],[0,0],[0,0],[0,0],[0,0],[1,-1]]
    v2=[[0,0],[0,1],[0,0],[1,-1],[0,0],[0,0]]
    
    divs = np.array([1,1,-1,-1,1,1])/2.0
        
    return v0, v1, v2, divs

def compute_nodal_tetrahedron():
    """
    The nodal vectors are computed using the following normals
        n0 = np.array([-1,-1, -1])/np.sqrt(3)
        n1 = np.array([-1, 0, 0])
        n2 = np.array([0, 1, 0])
        n3 = np.array([0, 0, -1])
    """
    

    v0 = [[0,0,0],[0,0,0],[0,0,0],[-1,0,0],[0,0,0],[0,0,0],\
         [0,1,0],[0,0,0],[0,0,0],[0,0,-1],[0,0,0],[0,0,0]]
    
    v1 = [[-1,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],\
         [0,0,0],[-1,1,0],[0,0,0],[0,0,0],[1,0,-1],[0,0,0]]
    
    v2 = [[0,0,0],[0,-1,0],[0,0,0],[0,0,0],[-1,1,0],[0,0,0],\
         [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,1,-1]]
    
    v3 = [[0,0,0],[0,0,0],[0,0,-1],[0,0,0],[0,0,0],[-1,0,1],\
         [0,0,0],[0,0,0],[0,1,-1],[0,0,0],[0,0,0],[0,0,0]]
    
    divs = np.array([-1,-1,-1,1,1,1,-1,-1,-1,1,1,1])/6.0
        
    return v0, v1, v2, v3, divs


def assemble_1d(mesh):
    DG = df.FunctionSpace(mesh, "DG", 0)
    n = df.FacetNormal(mesh)
    h = df.CellSize(mesh)
    h_avg = (h('+') + h('-'))/2
    
    u = df.TrialFunction(DG)
    v = df.TestFunction(DG)
    
    a = 1.0/h_avg*df.dot(df.jump(v, n), df.jump(u, n))*df.dS
    
    K = df.assemble(a)
    L = df.assemble(v * df.dx).array()
    
    return copy_petsc_to_csr(K), L

def assemble_2d(mesh):
    
    v0, v1, v2, divs = compute_nodal_triangle()

    cs = mesh.coordinates()
    BDM = df.FunctionSpace(mesh, "BDM", 1)
    fun = df.Function(BDM)
    n = fun.vector().size()
    mat = sp.lil_matrix((n,n))
    
    m = mesh.num_cells()
    mat_K = sp.lil_matrix((n,m))

    map = BDM.dofmap()
    
    for cell in df.cells(mesh):
        
        i = cell.entities(0)
        
        cm = []
        cm.append(cs[i[1]] - cs[i[0]])
        cm.append(cs[i[2]] - cs[i[0]])
        
        A = np.transpose(np.array(cm))
        B = np.dot(np.transpose(A),A)
        J = np.linalg.det(A)

        K = B/abs(J)        
        cfs = map.cell_dofs(cell.index())
            
        for i in range(6):
            for j in range(6):
                existing = mat[cfs[i],cfs[j]]
                add_new = np.dot(np.dot(K,v0[i]),v0[j]) \
                            + np.dot(np.dot(K,v1[i]),v1[j]) \
                            + np.dot(np.dot(K,v2[i]),v2[j])
                            
                mat[cfs[i],cfs[j]] = existing + add_new/6.0
                
        id_c = cell.index()
        for j in range(6):
            existing = mat_K[cfs[j],id_c]
            if J>0:
                mat_K[cfs[j],id_c] = existing + divs[j] 
            else:
                mat_K[cfs[j],id_c] = existing - divs[j]

    idy = generate_nonzero_ids(mat)
    #set the Neumann boundary condition here
    mesh.init(1,2)
    for edge in df.edges(mesh):
        faces = edge.entities(2)
        if len(faces)==1:
            f = df.Face(mesh,faces[0])
            cfs = map.cell_dofs(f.index())
            ids = map.tabulate_facet_dofs(f.index(edge))
            zid = cfs[ids]
            
            for i in zid:
                mat[i,idy[i]]=0
                mat[idy[i],i]=0
                mat[i,i] = 1
    

    A_inv = spl.inv(mat.tocsc())
    K3 = A_inv * mat_K.tocsr()

    K3 = K3.tolil()
    idy=generate_nonzero_ids(K3)
    mesh.init(1,2)
    for edge in df.edges(mesh):
        faces = edge.entities(2)
        if len(faces)==1:
            f = df.Face(mesh,faces[0])
            cfs = map.cell_dofs(f.index())
            ids = map.tabulate_facet_dofs(f.index(edge))
            for i in cfs[ids]:
                K3[i,idy[i]] = 0
                
    
    K1 = mat_K.transpose()
    K = K1*K3.tocsr()

    DG = df.FunctionSpace(mesh, "DG", 0)
    v = df.TestFunction(DG)

    L = df.assemble(v * df.dx).array()
    
    return K,L


def assemble_3d(mesh):
    
    v0, v1, v2, v3, divs = compute_nodal_tetrahedron()

    cs = mesh.coordinates()
    BDM = df.FunctionSpace(mesh, "BDM", 1)
    fun = df.Function(BDM)
    n = fun.vector().size()
    mat = sp.lil_matrix((n,n))
    
    m = mesh.num_cells()
    mat_K = sp.lil_matrix((n,m))

    map = BDM.dofmap()
    
    for cell in df.cells(mesh):
        
        ci = cell.entities(0)
        
        cm = []
        cm.append(cs[ci[1]] - cs[ci[0]])
        cm.append(cs[ci[2]] - cs[ci[0]])
        cm.append(cs[ci[3]] - cs[ci[0]])
        
        A = np.transpose(np.array(cm))
        B = np.dot(np.transpose(A),A)
        
        J = np.linalg.det(A)
        
        K = B/abs(J)
        
        cfs = map.cell_dofs(cell.index())
        
        for i in range(12):
            for j in range(12):
                tmp = mat[cfs[i],cfs[j]]
                tmp_res = np.dot(np.dot(K,v0[i]),v0[j]) \
                            + np.dot(np.dot(K,v1[i]),v1[j]) \
                            + np.dot(np.dot(K,v2[i]),v2[j]) \
                            + np.dot(np.dot(K,v3[i]),v3[j])
                mat[cfs[i],cfs[j]] = tmp + tmp_res/24.0
                
        id_c = cell.index()
        
        for j in range(12):
            tmp = mat_K[cfs[j],id_c]
            if J>0:
                mat_K[cfs[j],id_c] = tmp + divs[j]
            else:
                mat_K[cfs[j],id_c] = tmp - divs[j]

    idy=generate_nonzero_ids(mat)
    mesh.init(2,3)
    for face in df.faces(mesh):
        cells = face.entities(3)
        if len(cells)==1:
            c = df.Cell(mesh,cells[0])
            cfs = map.cell_dofs(c.index())
            ids = map.tabulate_facet_dofs(c.index(face))
            zid = cfs[ids]
            for i in zid:
                mat[i,idy[i]]=0
                mat[idy[i],i]=0
                mat[i,i] = 1
                
    import time
    t1=time.time()
    A_inv=sparse_inverse(mat.tocsc())
    #t2=time.time()
    #print 't2-t1 (s)',t2-t1
    #A_inv = spl.inv(mat.tocsc())
    #t3=time.time()
    #print 't3-t2 (s)',t3-t2
    K3 = A_inv * mat_K.tocsr()
    

    
    K3 = K3.tolil()
    idy=generate_nonzero_ids(K3)
    mesh.init(2,3)
    for face in df.faces(mesh):
        cells = face.entities(3)
        if len(cells)==1:
            c = df.Cell(mesh,cells[0])
            cfs = map.cell_dofs(c.index())
            ids = map.tabulate_facet_dofs(c.index(face))
            for i in cfs[ids]:
                K3[i,idy[i]] = 0
        
    K1 = mat_K.transpose()
    K = K1*K3.tocsr()

    DG = df.FunctionSpace(mesh, "DG", 0)
    v = df.TestFunction(DG)

    L = df.assemble(v * df.dx).array()
    
    return K,L

class ExchangeDG(object):
    def __init__(self, C, in_jacobian = False, name='ExchangeDG'):
        self.C = C
        self.in_jacobian=in_jacobian
        self.name = name
   
    #@mtimed
    def setup(self, DG3, m, Ms, unit_length=1.0): 
        self.DG3 = DG3
        self.m = m
        self.Ms = Ms
        self.unit_length = unit_length
        
        mesh = DG3.mesh()
        dim = mesh.topology().dim()
        
        if dim == 1:
            self.K, self.L = assemble_1d(mesh)
        elif dim == 2:
            self.K, self.L = assemble_2d(mesh)
        elif dim == 3:
            self.K, self.L = assemble_3d(mesh)
        
        self.mu0 = mu0
        self.exchange_factor = 2.0 * self.C / (self.mu0 * Ms * self.unit_length**2)
        
        self.coeff = -self.exchange_factor/self.L
        
        self.H = m.vector().array()
    
    def compute_field(self):
        mm = self.m.vector().array()
        mm.shape = (3,-1)
        self.H.shape=(3,-1)

        for i in range(3):
            self.H[i][:] = self.coeff * (self.K * mm[i])
        
        mm.shape = (-1,)
        self.H.shape=(-1,)
        
        return self.H
    
    def average_field(self):
        """
        Compute the average field.
        """
        return helpers.average_field(self.compute_field())



class ExchangeDG2(object):
    def __init__(self, C, in_jacobian = True, name='ExchangeDG'):
        self.C = C
        self.in_jacobian=in_jacobian
        self.name = name
   
    @mtimed
    def setup(self, DG3, m, Ms, unit_length=1.0): 
        self.DG3 = DG3
        self.m = m
        self.Ms = Ms
        self.unit_length = unit_length
        
        mesh = DG3.mesh()
        self.mesh = mesh
        
        DG = df.FunctionSpace(mesh, "DG", 0)
        BDM = df.FunctionSpace(mesh, "BDM", 1)
        
        #deal with three components simultaneously, each represents a vector
       
        sigma = df.TrialFunction(BDM)
        tau = df.TestFunction(BDM)
        
        
        u = df.TrialFunction(DG)
        v = df.TestFunction(DG)
        
        # what we need is A x = K1 m 
        #a0 = (df.dot(sigma0, tau0) + df.dot(sigma1, tau1) + df.dot(sigma2, tau2)) * df.dx
        a0 = df.dot(sigma, tau) * df.dx
        self.A = df.assemble(a0)
         
        a1 = - (df.div(tau) * u) * df.dx
        self.K1 = df.assemble(a1)
        
        C = sp.lil_matrix(self.K1.array())
        self.KK1 = C.tocsr()
    
        
        def boundary(x, on_boundary):
            return on_boundary
        
        # actually, we need to apply the Neumann boundary conditions.
        
        zero = df.Constant((0,0,0))
        self.bc = df.DirichletBC(BDM, zero, boundary)
        #print 'before',self.A.array()
        
        self.bc.apply(self.A)
        
        #print 'after',self.A.array()
        
        #AA = sp.lil_matrix(self.A.array())
        AA = copy_petsc_to_csc(self.A)
        
        self.solver = sp.linalg.factorized(AA.tocsc())
        
        #LU = sp.linalg.spilu(AA)
        #self.solver = LU.solve
        
        a2 = (df.div(sigma) * v) * df.dx
        self.K2 = df.assemble(a2)
        self.L = df.assemble(v * df.dx).array()
    
        self.mu0 = mu0
        self.exchange_factor = 2.0 * self.C / (self.mu0 * Ms * self.unit_length**2)

        self.coeff = self.exchange_factor/self.L
        
        self.K2 = copy_petsc_to_csr(self.K2)
        
        # b = K m
        self.b = df.PETScVector()        
        # the vector in BDM space
        self.sigma_v = df.PETScVector()
        
        # to store the exchange fields
        #self.H = df.PETScVector()
        self.H_eff = m.vector().array()
        
        self.m_x = df.PETScVector(self.m.vector().size()/3)
        
    
    @mtimed
    def compute_field(self):
        
        mm = self.m.vector().array()
        mm.shape = (3,-1)
        self.H_eff.shape=(3,-1)
        
        for i in range(3):
            self.m_x.set_local(mm[i])
            self.K1.mult(self.m_x, self.b)
            self.bc.apply(self.b)      
            
            H = self.solver(self.b.array())
            #df.solve(self.A, self.sigma_v, self.b)
            
            self.H_eff[i][:] = (self.K2*H)*self.coeff
        
        mm.shape = (-1,)
        self.H_eff.shape=(-1,)
        
        return self.H_eff
        
    
    def average_field(self):
        """
        Compute the average field.
        """
        return helpers.average_field(self.compute_field())


"""
Compute the exchange field in DG0 space with the help of BDM1 space.
""" 
class ExchangeDG_bak(object):
    def __init__(self, C, in_jacobian = False, name='ExchangeDG'):
        self.C = C
        self.in_jacobian=in_jacobian
        self.name = name
   
    @mtimed
    def setup(self, DG3, m, Ms, unit_length=1.0): 
        self.DG3 = DG3
        self.m = m
        self.Ms = Ms
        self.unit_length = unit_length
        
        mesh = DG3.mesh()
        self.mesh = mesh
        
        DG = df.FunctionSpace(mesh, "DG", 0)
        BDM = df.FunctionSpace(mesh, "BDM", 1)
        
        #deal with three components simultaneously, each represents a vector
        W1 = df.MixedFunctionSpace([BDM, BDM, BDM])
        (sigma0,sigma1,sigma2) = df.TrialFunctions(W1)
        (tau0,tau1,tau2) = df.TestFunctions(W1)
        
        W2 = df.MixedFunctionSpace([DG, DG, DG])
        (u0,u1,u2) = df.TrialFunctions(W2)
        (v0,v1,v2) = df.TestFunction(W2)
        
        # what we need is A x = K1 m 
        a0 = (df.dot(sigma0, tau0) + df.dot(sigma1, tau1) + df.dot(sigma2, tau2)) * df.dx
        self.A = df.assemble(a0)
         
        a1 = - (df.div(tau0) * u0 + df.div(tau1) * u1 + df.div(tau2) * u2 ) * df.dx
        self.K1 = df.assemble(a1)
    
        
        def boundary(x, on_boundary):
            return on_boundary
        
        # actually, we need to apply the Neumann boundary conditions.
        # we need a tensor here
        zero = df.Constant((0,0,0,0,0,0,0,0,0))
        self.bc = df.DirichletBC(W1, zero, boundary)
        self.bc.apply(self.A)
        
        
        a2 = (df.div(sigma0) * v0 + df.div(sigma1) * v1 + df.div(sigma2) * v2) * df.dx
        self.K2 = df.assemble(a2)
        self.L = df.assemble((v0 + v1 + v2) * df.dx).array()
    
        self.mu0 = mu0
        self.exchange_factor = 2.0 * self.C / (self.mu0 * Ms * self.unit_length**2)

        self.coeff = self.exchange_factor/self.L
        
        # b = K m
        self.b = df.PETScVector()
        
        # the vector in BDM space
        self.sigma_v = df.PETScVector(self.K2.size(1))
        
        # to store the exchange fields
        self.H = df.PETScVector()
    
    @mtimed
    def compute_field(self):
        
        # b = K2 * m 
        self.K1.mult(self.m.vector(), self.b)
        
        self.bc.apply(self.b)
        
        df.solve(self.A, self.sigma_v, self.b)
        
        self.K2.mult(self.sigma_v, self.H)

        return self.H.array()*self.coeff
    
    def average_field(self):
        """
        Compute the average field.
        """
        return helpers.average_field(self.compute_field())
 
