import dolfin as df
import numpy as np
df.parameters.reorder_dofs_serial = False

mu0=np.pi*4e-7

#import matplotlib as mpl
#mpl.use("Agg")
#import matplotlib.pyplot as plt


def compute_phi(mesh,m):
    V = df.FunctionSpace(mesh, 'CG', 1)
    W = df.VectorFunctionSpace(mesh, 'CG', 1)
    w = df.TrialFunction(W)
    v = df.TestFunction(V)
    b = df.inner(w, df.grad(v))*df.dx
    D = df.assemble(b)
    n = df.FacetNormal(mesh)
    print 'cg',df.assemble(df.dot(w,n)*v*df.ds).array()
    h = D * m.vector()
    
    f = df.Function(V)
    f.vector().set_local(h.array())
    return f


def compute_phi2(mesh,m):
    V = df.FunctionSpace(mesh, 'CG', 1)
    W = df.VectorFunctionSpace(mesh, 'CG', 1)
    w = df.TrialFunction(W)
    v = df.TestFunction(V)
    n = df.FacetNormal(mesh)
    b = -df.div(w)*v*df.dx+df.dot(w,n)*v*df.ds
    D = df.assemble(b)
    print '='*100,'CG:', np.max(df.assemble(df.div(w)*v*df.dx).array())
    
    h = D * m.vector()

    f = df.Function(V)
    f.vector().set_local(h.array())
    return f

def compute_phi_dg(mesh,m):
    V = df.FunctionSpace(mesh, 'CG', 1)
    W = df.VectorFunctionSpace(mesh, 'DG', 0)
    w = df.TrialFunction(W)
    v = df.TestFunction(V)
    n = df.FacetNormal(mesh)
    b = df.inner(w, df.grad(v))*df.dx
    D = df.assemble(b)
        
    h = D * m.vector()
    f = df.Function(V)
    f.vector().set_local(h.array())
    return f

def compute_phi_dg2(mesh,m):
    V = df.FunctionSpace(mesh, 'CG', 1)
    W = df.VectorFunctionSpace(mesh, 'DG', 0)
    w = df.TrialFunction(W)
    v = df.TestFunction(V)
    n = df.FacetNormal(mesh)
    b = -df.div(w)*v*df.dx+df.dot(w,n)*v*df.ds
    D = df.assemble(b)
    print '='*100,  np.max(df.assemble(df.div(w)*v*df.dx).array())
    h = D * m.vector()

    f = df.Function(V)
    f.vector().set_local(h.array())
    return f

def compute_phi_direct(mesh,m):
    V = df.FunctionSpace(mesh, 'CG', 1)
    n = df.FacetNormal(mesh)
    v = df.TestFunction(V)
    b = df.dot(n, m)*v*df.ds - df.div(m)*v*df.dx
    h = df.assemble(b)

    f = df.Function(V)
    f.vector().set_local(h.array())
    return f


def test_phi(mesh):
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    Vd = df.VectorFunctionSpace(mesh, 'DG', 0)
    expr = df.Expression(("sin(x[0])","cos(x[0])","0"))
    m_cg = df.interpolate(expr, V)
    m_dg = df.interpolate(expr, Vd)
    
    p_cg=compute_phi(mesh,m_cg)
    p_cg2=compute_phi2(mesh,m_cg)
    p_cg3=compute_phi_direct(mesh,m_cg)
    p_dg=compute_phi_dg(mesh,m_dg)
    p_dg2=compute_phi_dg2(mesh,m_dg)
    p_dg3=compute_phi_direct(mesh,m_dg)

    d = p_dg.vector().array()-p_dg2.vector().array()
    print 'dg max diff',np.max(abs(d))
    d = p_dg2.vector().array()-p_dg3.vector().array()
    print 'dg max2 diff',np.max(abs(d))
    d = p_cg.vector().array()-p_cg2.vector().array()
    print 'cg max diff',np.max(abs(d))
    d = p_cg2.vector().array()-p_cg3.vector().array()
    print 'cg max2 diff',np.max(abs(d))
    df.plot(p_cg)
    df.plot(p_cg2)
    df.plot(p_dg)
    df.plot(p_dg2)

def laplace(mesh,phi):
    S1 = df.FunctionSpace(mesh, "CG", 1)
    S3 = df.VectorFunctionSpace(mesh, "CG", 1)
    
    u1 = df.TrialFunction(S1)
    v1 = df.TestFunction(S1)
	
    u3 = df.TrialFunction(S3)
    v3 = df.TestFunction(S3)
	
    K = df.assemble(df.inner(df.grad(u1), v3) * df.dx)
    L = df.assemble(v1 * df.dx)
    
    h = K*phi.vector() 
    
    f = df.Function(S3)
    f.vector().set_local(h.array())

    return f


def laplace_dg(mesh,phi):
    S1 = df.FunctionSpace(mesh, "CG", 1)
    S3 = df.VectorFunctionSpace(mesh, "DG", 0)

    u1 = df.TrialFunction(S1)
    v1 = df.TestFunction(S1)

    u3 = df.TrialFunction(S3)
    v3 = df.TestFunction(S3)

    K = df.assemble(df.inner(df.grad(u1), v3) * df.dx)
    L = df.assemble(v1 * df.dx)

    h = K*phi.vector()

    f = df.Function(S3)
    f.vector().set_local(h.array())

    return f


if __name__ == "__main__":
   
    mesh = df.UnitCubeMesh(32,32,2) 
    S = df.FunctionSpace(mesh, "CG", 1)
    
    expr = df.Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.2)")
    phi = df.interpolate(expr, S)
    #df.plot(phi)
    
    m=laplace(mesh,phi)
    m_dg=laplace_dg(mesh,phi)
    #df.plot(m_dg)

    test_phi(mesh)
    
    #df.interactive()
    
    
    

    
    
    

    
    



    
