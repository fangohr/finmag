import dolfin as df
import numpy as np
#from finmag.energies.exchange import Exchange 
#from finmag.util.consts import mu0
mu0=np.pi*4e-7





def exchange(mesh,m,dim=2,C=1.3e-11,Ms=8.6e5):
    V = df.FunctionSpace(mesh, "Lagrange", 1)
    u1 = df.TrialFunction(V)
    v1 = df.TestFunction(V)
    K = df.assemble(df.inner(df.grad(u1), df.grad(v1)) * df.dx).array()
    L = df.assemble(v1 * df.dx).array()
    
    print K
    print L
   
    m.shape=(dim,-1)
    h=np.zeros(m.shape)
    
    for i in range(dim):
        h[i][:]=-np.dot(K,m[i])/L*(2*C)/(mu0*Ms)
    m.shape=(-1)
    h.shape=(-1)
    return h


class PeriodicBoundary_X(df.SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < df.DOLFIN_EPS and x[0] > -df.DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]
        

class PeriodicBoundary_Y(df.SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[1] < df.DOLFIN_EPS and x[1] > -df.DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1] - 1.0


def periodic_exchange(mesh,m,dim=2,C=1.3e-11,Ms=8.6e5):
    V = df.FunctionSpace(mesh, "Lagrange", 1)
    u1 = df.TrialFunction(V)
    v1 = df.TestFunction(V)
    K = df.assemble(df.inner(df.grad(u1), df.grad(v1)) * df.dx)
    L = df.assemble(v1 * df.dx)
    
    bcx = df.PeriodicBC(V, PeriodicBoundary_X())
    bcy = df.PeriodicBC(V, PeriodicBoundary_Y())
    
    print K.array()
    print L.array()
    bcx.apply(K,L)
    #bcy.apply(K,L)
    print K.array()
    print L.array()
   
    m.shape=(dim,-1)
    h=np.zeros(m.shape)
    K=K.array()
    L=L.array()
    
    for i in range(dim):
        h[i][:]=-np.dot(K,m[i])/L*(2*C)/(mu0*Ms)
    m.shape=(-1)
    h.shape=(-1)
    return h


def exchange_vector(mesh,m,dim=2,C=1.3e-11,Ms=8.6e5):
    V3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    u3 = df.TrialFunction(V3)
    v3 = df.TestFunction(V3)
    K = df.assemble(df.inner(df.grad(u3),df.grad(v3))* df.dx)
    L = df.assemble(df.dot(v3, df.Constant([1, 1])) * df.dx)
    
    bcx = df.PeriodicBC(V3, PeriodicBoundary_X())
    bcy = df.PeriodicBC(V3, PeriodicBoundary_Y())
    
    print L.array()
    bcx.apply(K,L)
    print L.array()
    
    K=K.array()
    L=L.array()
    
    h=-np.dot(K,m)/L*(2*C)/(mu0*Ms)
    
    return h

if __name__ == "__main__":
    dim=2
    mesh = df.UnitSquareMesh(2, 2)  
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=dim)
    C = 1
    expr = df.Expression(('sin(x[0])', 'cos(x[0])'))
    Ms = 1
    M = df.project(expr, S3)
    df.plot(mesh)
    df.interactive()
    
    #print exchange(mesh,M.vector().array(),dim,C,Ms)
    print exchange_vector(mesh,M.vector().array(),dim,C,Ms)
    
    
    

    
    



    