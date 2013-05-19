import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import finmag
import dolfin as df
import numpy as np



xs=np.linspace(1e-10,2*np.pi,10)

def delta_cg(mesh,expr):
    V = df.FunctionSpace(mesh, "CG", 1)
    Vv = df.VectorFunctionSpace(mesh, "CG", 1)
    u = df.interpolate(expr, Vv)
    
    u1 = df.TrialFunction(Vv)
    v1 = df.TestFunction(Vv)
    K = df.assemble(df.inner(df.grad(u1), df.grad(v1)) * df.dx).array()
    L = df.assemble(df.dot(v1, df.Constant([1,1,1])) * df.dx).array()
    
    h=-np.dot(K,u.vector().array())/L
        
    fun = df.Function(Vv)
    fun.vector().set_local(h)
    
    res = []
    for x in xs:
        res.append(fun(x,5,0.5)[0])
    df.plot(fun)
    return res


def delta_dg(mesh,expr):
    CG = df.FunctionSpace(mesh, "CG", 1)
    DG = df.FunctionSpace(mesh, "DG", 0)
    
    CG3 = df.VectorFunctionSpace(mesh, "CG", 1)
    DG3 = df.VectorFunctionSpace(mesh, "DG", 0)
    
    m = df.interpolate(expr, DG3)
    n = df.FacetNormal(mesh)
    
    
    u_dg = df.TrialFunction(DG)
    v_dg = df.TestFunction(DG)
    
    u3 = df.TrialFunction(CG3)
    v3 = df.TestFunction(CG3)
        
    a = u_dg * df.inner(v3, n) * df.ds - u_dg * df.div(v3) * df.dx
    
    mm = m.vector().array()
    mm.shape = (3,-1)
    
    K = df.assemble(a).array()
    L3 = df.assemble(df.dot(v3, df.Constant([1,1,1])) * df.dx).array()
    f = np.dot(K,mm[0])/L3
    
    fun1 = df.Function(CG3)
    fun1.vector().set_local(f)
    df.plot(fun1)
    
    
    a = df.div(u3) * v_dg * df.dx
    A = df.assemble(a).array()
    L = df.assemble(v_dg * df.dx).array()
    
    h =  np.dot(A,f)/L
    
    fun = df.Function(DG)
    fun.vector().set_local(h)
    df.plot(fun)
    res = []
    for x in xs:
        res.append(fun(x,5,0.5))
    
    """
    fun2 =df.interpolate(fun, df.VectorFunctionSpace(mesh, "CG", 1))
    file = df.File('field2.pvd')
    file << fun2
    """
    
    
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
    
    mesh = df.BoxMesh(0,0,0,2*np.pi,10,1,10, 10, 1)
    #mesh = df.IntervalMesh(10, 0, 2*np.pi)
    
    expr = df.Expression('cos(x[0])')
    
    expr = df.Expression(('cos(x[0])','0','0'))
    m_an=-np.cos(xs)
    plot_m(mesh, expr, m_an, name='cos_3d_2.pdf')
    
    """
    expr = df.Expression(('sin(x[0])','0','0'))
    m_an=-np.sin(xs)
    plot_m(mesh, expr, m_an, name='sin_3d_2.pdf')
    
    
    expr = df.Expression(('x[0]*x[0]','0','0'))
    m_an=2+1e-10*xs
    plot_m(mesh, expr, m_an, name='x2_3d_2.pdf')
    """
    
    
    #df.interactive()