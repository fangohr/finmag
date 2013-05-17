import dolfin as df
import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def delta_cg(mesh,expr):
    V = df.FunctionSpace(mesh, "CG", 1)
    u = df.interpolate(expr, V)
    
    u1 = df.TrialFunction(V)
    v1 = df.TestFunction(V)
    K = df.assemble(df.inner(df.grad(u1), df.grad(v1)) * df.dx).array()
    L = df.assemble(v1 * df.dx).array()
    
    h=-np.dot(K,u.vector().array())/L
    
    return h


def delta_dg(mesh,expr):
    V = df.FunctionSpace(mesh, "DG", 0)
    m = df.interpolate(expr, V)
    
    n = df.FacetNormal(mesh)
    h = df.CellSize(mesh)
    h_avg = (h('+') + h('-'))/2
    
    alpha = 1
    gamma = 0
    
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    
    # for DG 0 case, only term contain alpha is nonzero
    a = df.dot(df.grad(v), df.grad(u))*df.dx \
        - df.dot(df.avg(df.grad(v)), df.jump(u, n))*df.dS \
        - df.dot(df.jump(v, n), df.avg(df.grad(u)))*df.dS \
        + alpha/h_avg*df.dot(df.jump(v, n), df.jump(u, n))*df.dS \
        - df.dot(df.grad(v), u*n)*df.ds \
        - df.dot(v*n, df.grad(u))*df.ds \
        + gamma/h*v*u*df.ds
    
    K = df.assemble(a).array()
    L = df.assemble(v * df.dx).array()
    
    h = -np.dot(K,m.vector().array())/L
    
    xs=[]
    for cell in df.cells(mesh):
        xs.append(cell.midpoint().x())
    
    print len(xs),len(h)
    return xs,h


def plot_m(mesh,expr,m_an,name='compare.pdf'):
    fig=plt.figure()
    
    xs=mesh.coordinates().flatten()

    plt.plot(xs,m_an,'--',label='Analytical')
    
    cg = delta_cg(mesh,expr)
    plt.plot(xs,cg,'.-',label='cg')
    
    xs,dg = delta_dg(mesh,expr)
    print xs,dg
    plt.plot(xs,dg,'^--',label='dg')

    plt.legend(loc=8)
    fig.savefig(name)


if __name__ == "__main__":
    
    mesh = df.UnitIntervalMesh(10)
    xs=mesh.coordinates().flatten()
    
    expr = df.Expression('cos(2*pi*x[0])')
    m_an=-(2*np.pi)**2*np.cos(2*np.pi*xs)
    plot_m(mesh, expr, m_an, name='cos.pdf')
    
    expr = df.Expression('sin(2*pi*x[0])')
    m_an=-(2*np.pi)**2*np.sin(2*np.pi*xs)
    plot_m(mesh, expr, m_an, name='sin.pdf')
    
    expr = df.Expression('x[0]*x[0]')
    m_an=2+1e-10*xs
    plot_m(mesh, expr, m_an, name='x2.pdf') 
    
