import dolfin as df
import numpy as np
df.parameters.reorder_dofs_serial = False
#from finmag.energies.exchange import Exchange 
#from finmag.util.consts import mu0
mu0=np.pi*4e-7

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


class PeriodicBoundary2D(df.SubDomain):

    def inside(self, x, on_boundary):
        
        on_x = bool(df.near(x[0],0) and not df.near(x[1],1.0) and on_boundary)
        on_y = bool(df.near(x[1],0) and not df.near(x[0],1.0) and on_boundary)
        
        return on_x or on_y   
    

    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1]
        if df.near(x[0],1):
            y[0] = x[0] - 1
        
        if df.near(x[1],1):
            y[1] = x[1] - 1.0
        
        
        

def exchange(mesh,m):
    V = df.FunctionSpace(mesh, "Lagrange", 1)
    u1 = df.TrialFunction(V)
    v1 = df.TestFunction(V)
    K = df.assemble(df.inner(df.grad(u1), df.grad(v1)) * df.dx).array()
    L = df.assemble(v1 * df.dx).array()
    
    h=-np.dot(K,m)/L
    
    print K.shape,'\n',K
    print 'L=',L
    print 'm',m

    return h

def periodic_exchange(mesh,m):
    V = df.FunctionSpace(mesh, "Lagrange", 1, constrained_domain=PeriodicBoundary2D())
    u1 = df.TrialFunction(V)
    v1 = df.TestFunction(V)
    K = df.assemble(df.inner(df.grad(u1), df.grad(v1)) * df.dx).array()
    print K.shape,'\n',K
    L = df.assemble(v1 * df.dx).array()
    print L
    print m
    
    h=-np.dot(K,m)/L
    #h[-1]=h[0]

    return h


if __name__ == "__main__":
   
    
    mesh = df.UnitSquareMesh(3,2) 
    S = df.FunctionSpace(mesh, "Lagrange", 1)
    
    
    expr = df.Expression('cos(2*pi*x[0])')
    M = df.interpolate(expr, S)
    df.plot(M)
    #df.interactive()
    
    
    S2 = df.FunctionSpace(mesh, "Lagrange", 1, constrained_domain=PeriodicBoundary2D())
    M2 = df.interpolate(expr, S2)
    
    print exchange(mesh,M.vector().array())
    print '='*100
    print periodic_exchange(mesh,M2.vector().array())
    
    
    
    

    
    
    

    
    



    