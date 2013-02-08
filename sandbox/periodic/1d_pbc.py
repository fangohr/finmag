import dolfin as df
import numpy as np
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

class PeriodicBoundary2(df.SubDomain):
    
    def inside(self, x, on_boundary):
        return df.near(x[0],1.0) and on_boundary

    def map(self, x, y):
        y[0] = x[0] + 1.0


def exchange(mesh,m):
    V = df.FunctionSpace(mesh, "Lagrange", 1)
    u1 = df.TrialFunction(V)
    v1 = df.TestFunction(V)
    K = df.assemble(df.inner(df.grad(u1), df.grad(v1)) * df.dx).array()
    L = df.assemble(v1 * df.dx).array()
    
    h=-np.dot(K,m)/L

    return h

def periodic_exchange(mesh,m):
    V = df.FunctionSpace(mesh, "Lagrange", 1)
    u1 = df.TrialFunction(V)
    v1 = df.TestFunction(V)
    K = df.assemble(df.inner(df.grad(u1), df.grad(v1)) * df.dx)
    L = df.assemble(v1 * df.dx)
    
    bc1 = df.PeriodicBC(V, PeriodicBoundary1())
    
    bc1.apply(K,L)
    
    K=K.array()
    L=L.array()
    L[-1]=L[0]

    h=-np.dot(K,m)/L
    h[-1]=h[0]

    return h



def plot_m(mesh,m,xs,m_an,name):
    
    fig=plt.figure()
    plt.plot(xs,m_an,label='Analytical')
    openbc=exchange(mesh,m)
    plt.plot(xs,openbc,'.-',label='OpenBC')
    pbc=periodic_exchange(mesh,m)
    plt.plot(xs,pbc,'^-',label='PBC')
    plt.legend()
    fig.savefig(name)



if __name__ == "__main__":
   
    
    mesh = df.UnitIntervalMesh(10) 
    S = df.FunctionSpace(mesh, "Lagrange", 1)

    expr = df.Expression('cos(2*pi*x[0])')
    M = df.interpolate(expr, S)
    #df.plot(mesh)
    #df.interactive()
    
    xs=mesh.coordinates().flatten()
    m_an=-(2*np.pi)**2*np.cos(2*np.pi*xs)
    plot_m(mesh,M.vector().array(),xs,m_an,'1d_cos.png')
    
    expr = df.Expression('sin(2*pi*x[0])')
    M = df.interpolate(expr, S)
    m_an=-(2*np.pi)**2*np.sin(2*np.pi*xs)
    plot_m(mesh,M.vector().array(),xs,m_an,'1d_sin.png')
    
    
    

    
    



    