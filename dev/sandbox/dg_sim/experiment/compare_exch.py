import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import finmag
import dolfin as df
import numpy as np



xs=np.linspace(1e-10+0.5,2*np.pi-0.5,10)

def delta_cg(mesh,expr):
    V = df.FunctionSpace(mesh, "CG", 1)
    u = df.interpolate(expr, V)
    
    u1 = df.TrialFunction(V)
    v1 = df.TestFunction(V)
    K = df.assemble(df.inner(df.grad(u1), df.grad(v1)) * df.dx).array()
    L = df.assemble(v1 * df.dx).array()
    
    h=-np.dot(K,u.vector().array())/L
    
    fun = df.Function(V)
    fun.vector().set_local(h)
    
    dim = mesh.topology().dim()
    
    res = []    
    
    if dim == 1:
        for x in xs:
            res.append(fun(x))
    elif dim == 2:
        df.plot(fun)
        #df.interactive()
        for x in xs:
            res.append(fun(x,0.5))
    elif dim == 3:
        for x in xs:
            res.append(fun(x,0.5,0.5))
    
    return res

def printaa(expr,name):
    print '='*100
    print name
    print np.max(np.abs(df.assemble(expr).array()))
    print '-'*100


def delta_dg(mesh,expr):
    V = df.FunctionSpace(mesh, "DG", 1)
    m = df.interpolate(expr, V)
    
    n = df.FacetNormal(mesh)
    h = df.CellSize(mesh)
    h_avg = (h('+') + h('-'))/2.0
    
    alpha = 1.0
    gamma = 0
    
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    
    
    a = df.dot(df.grad(v), df.grad(u))*df.dx \
        - df.dot(df.avg(df.grad(v)), df.jump(u, n))*df.dS \
        - df.dot(df.jump(v, n), df.avg(df.grad(u)))*df.dS \
        + alpha/h_avg*df.dot(df.jump(v, n), df.jump(u, n))*df.dS 
        #- df.dot(df.grad(v), u*n)*df.ds \
        #- df.dot(v*n, df.grad(u))*df.ds \
        #+ gamma/h*v*u*df.ds
        
    #a = 1.0/h_avg*df.dot(df.jump(v, n), df.jump(u, n))*df.dS 
        
        #- df.dot(df.grad(v), u*n)*df.ds \
        #- df.dot(v*n, df.grad(u))*df.ds \
        #+ gamma/h*v*u*df.ds

    """
    a1 = df.dot(df.grad(v), df.grad(u))*df.dx 
    a2 = df.dot(df.avg(df.grad(v)), df.jump(u, n))*df.dS
    a3 = df.dot(df.jump(v, n), df.avg(df.grad(u)))*df.dS
    a4 = alpha/h_avg*df.dot(df.jump(v, n), df.jump(u, n))*df.dS 
    a5 = df.dot(df.grad(v), u*n)*df.ds
    a6 = df.dot(v*n, df.grad(u))*df.ds
    a7 = alpha/h*v*u*df.ds
    
    printaa(a1,'a1')
    printaa(a2,'a2')
    printaa(a3,'a3')
    printaa(a4,'a4')
    printaa(a5,'a5')
    printaa(a6,'a6')
    printaa(a7,'a7')
    """
    
    K = df.assemble(a).array()
    L = df.assemble(v * df.dx).array()
    
    
    h = -np.dot(K,m.vector().array())/L
    
    fun = df.Function(V)
    fun.vector().set_local(h)
    
    DG1 = df.FunctionSpace(mesh, "DG", 1)
    CG1 = df.FunctionSpace(mesh, "CG", 1)
    #fun = df.interpolate(fun, DG1)
    fun = df.project(fun, CG1)
    
    dim = mesh.topology().dim()
    
    res = []    
    
    if dim == 1:
        for x in xs:
            res.append(fun(x))
    elif dim == 2:
        df.plot(fun)
        df.interactive()
        for x in xs:
            res.append(fun(x,0.5))
    elif dim == 3:
        for x in xs:
            res.append(fun(x,0.5,0.5))
            
    return res


def plot_m(mesh,expr,m_an,xs=xs,name='compare.pdf'):
    
    fig=plt.figure()
    
    plt.plot(xs,m_an,'--',label='Analytical')
    
    cg = delta_cg(mesh,expr)
    plt.plot(xs,cg,'.-',label='cg')
    
    dg = delta_dg(mesh,expr)

    plt.plot(xs,dg,'^--',label='dg')

    plt.legend(loc=8)
    fig.savefig(name)


if __name__ == "__main__":
    
    
    expr = df.Expression('cos(x[0])')
    m_an=-np.cos(xs)
    
    
    mesh = df.IntervalMesh(100, 0, 2*np.pi)
    plot_m(mesh, expr, m_an, name='cos_1d.pdf')
    
    mesh = df.RectangleMesh(0,0,2*np.pi, 1, 20, 1)
    plot_m(mesh, expr, m_an, name='cos_2d.pdf')
    
    mesh = df.BoxMesh(0,0,0,2*np.pi,1,1,50, 1, 1)
    plot_m(mesh, expr, m_an, name='cos_3d.pdf')
    
    
    
    
    """
    expr = df.Expression('sin(x[0])')
    m_an=-np.sin(xs)
    plot_m(mesh, expr, m_an, name='sin_3d.pdf')
    
    expr = df.Expression('x[0]*x[0]')
    m_an=2+1e-10*xs
    plot_m(mesh, expr, m_an, name='x2_3d.pdf') 
    """