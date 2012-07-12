import dolfin as df
import numpy as np
from finmag.energies.exchange import Exchange 
from finmag.util.consts import mu0


def exchange(mesh,m,dim=2,C=1.3e-11,Ms=8.6e5):
    V = df.FunctionSpace(mesh, "Lagrange", 1)
    u1 = df.TrialFunction(V)
    v1 = df.TestFunction(V)
    K = df.assemble(df.inner(df.grad(u1), df.grad(v1)) * df.dx).array()
    L = df.assemble(v1 * df.dx).array()
   
    m.shape=(dim,-1)
    h=np.zeros(m.shape)
    
    for i in range(dim):
        h[i][:]=-np.dot(K,m[i])/L*(2*C)/(mu0*Ms)
    m.shape=(-1)
    h.shape=(-1)
    return h
    

if __name__ == "__main__":
    mesh = df.Interval(2, 0, 1)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=2)
    C = 1
    expr = df.Expression(('sin(x[0])', 'cos(x[0])'))
    Ms = 1
    M = df.project(expr, S3)
    
    exch = Exchange(C)

    exch.setup(S3, M, Ms)

    print exch.compute_field()
    print exchange(mesh,M.vector().array(),2,C,Ms)
    
    

    
    



    
