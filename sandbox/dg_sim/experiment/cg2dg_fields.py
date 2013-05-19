"""
The functions below are intend to test how to obtain the fields in DG0 space
for the case that the potential is stored in CG space. 
"""

import dolfin as df
import numpy as np
df.parameters.reorder_dofs_serial = False

def compute_field1(mesh,phi):
    V = df.FunctionSpace(mesh, 'CG', 1)
    DG3 = df.VectorFunctionSpace(mesh, 'DG', 0)

    u = df.TrialFunction(V)
    v = df.TestFunction(DG3)
   
    a = df.inner(df.grad(u), v) * df.dx
    A = df.assemble(a)
    L = df.assemble(df.dot(v, df.Constant([1,1])) * df.dx).array()
    
    m = A*phi.vector()
    
    f = df.Function(DG3)
    f.vector().set_local(m.array()/L)
    
    return f


def compute_field2(mesh,phi):
    V = df.FunctionSpace(mesh, 'CG', 1)
    DG3 = df.VectorFunctionSpace(mesh, 'DG', 0)

    sig = df.TrialFunction(DG3)
    v = df.TestFunction(DG3)
   
    a = df.inner(sig, v) * df.dx
    b = df.inner(df.grad(phi), v) * df.dx
    
    f = df.Function(DG3)
    df.solve(a == b, f)
    
    return f


if __name__ == "__main__":
   
    #mesh = df.UnitCubeMesh(32,32,1) 
    mesh = df.UnitSquareMesh(32,32) 
    S = df.FunctionSpace(mesh, "CG", 1)
    
    expr = df.Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.2)")
    phi = df.interpolate(expr, S)
    
    m1=compute_field1(mesh,phi)
    m2=compute_field2(mesh,phi)
    
    diff = m1.vector()-m2.vector()
    print 'max diff:',np.max(np.abs(diff.array()))
    
    df.plot(m1)
    df.plot(m2)

    df.interactive()
    
    
    

    
    
    

    
    



    
