import dolfin as df
import numpy as np
df.parameters.reorder_dofs_serial = False
#from finmag.energies.exchange import Exchange 
#from finmag.util.consts import mu0
mu0=np.pi*4e-7

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


class PeriodicBoundary1(df.SubDomain):
    
    def inside(self, x, on_boundary):
        return df.near(x[0],0) and on_boundary

    def map(self, x, y):
        y[0] = x[0] - 1.0



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
    V = df.FunctionSpace(mesh, "Lagrange", 1, constrained_domain=PeriodicBoundary1())
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



def plot_m(mesh,m,xs,m_an,m2,name):
    
    fig=plt.figure()
    plt.plot(xs,m_an,label='Analytical')
    openbc=exchange(mesh,m)
    plt.plot(xs,openbc,'--',label='OpenBC')
    pbc=periodic_exchange(mesh,m2)
    plt.plot(xs[:-1],pbc,'^',label='PBC')
    plt.legend()
    fig.savefig(name)



if __name__ == "__main__":
   
    
    mesh = df.UnitIntervalMesh(10) 
    S = df.FunctionSpace(mesh, "Lagrange", 1)

    expr = df.Expression('cos(2*pi*x[0])')
    M = df.interpolate(expr, S)
    #df.plot(mesh)
    #df.interactive()
    
    S2 = df.FunctionSpace(mesh, "Lagrange", 1, constrained_domain=PeriodicBoundary1())
    M2 = df.interpolate(expr, S2)
    
    xs=mesh.coordinates().flatten()
    m_an=-(2*np.pi)**2*np.cos(2*np.pi*xs)
    plot_m(mesh,M.vector().array(),xs,m_an,M2.vector().array(),'1d_cos.png')
    

    
    
    

    
    



    
