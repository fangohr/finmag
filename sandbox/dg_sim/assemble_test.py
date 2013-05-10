import os
import finmag
import dolfin as df
import numpy as np

    
def dg_cg(mesh):
    pass
    

if __name__ == "__main__":
    mesh = df.BoxMesh(0, 0, 0, np.pi, 1, 1, 10, 1, 1)
    
    CG = df.FunctionSpace(mesh, "CG", 1)
    DG = df.FunctionSpace(mesh, "DG", 0)
    CG3 = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    DG3 = df.VectorFunctionSpace(mesh, "DG", 0, dim=3)
    
    CG3_vector=df.dot(df.TestFunction(CG3), df.Constant([1, 1, 1]))
    
    
    expr = df.Expression(('sin(x[0])', 'cos(x[0])','0'))
    m_cg = df.interpolate(expr, CG3)
    m_dg = df.interpolate(expr, DG3)
    
    m_cg_arr = m_cg.vector().array()
    m_cg_arr.shape=(3,-1)
   
    print 'length1',np.sqrt(m_cg_arr[0]**2+m_cg_arr[1]**2+m_cg_arr[2]**2)
    
    m_int=df.interpolate(m_dg,CG3)
    m_int_arr = m_int.vector().array()
    m_int_arr.shape=(3,-1)

    print 'length2',np.sqrt(m_int_arr[0]**2+m_int_arr[1]**2+m_int_arr[2]**2)
    df.plot(m_int)
    
    m_int_arr/=np.sqrt(m_int_arr[0]**2+m_int_arr[1]**2+m_int_arr[2]**2)
    
    m_int_arr.shape=(-1,)
    print m_cg.vector().array()-m_int_arr

    mx = df.interpolate(df.Expression('sin(x[0])'), DG)
    my = df.interpolate(df.Expression('cos(x[0])'), DG)
    mz = df.interpolate(df.Expression('0'), DG)
    
    mnx = df.assemble(mx*df.TestFunction(CG) * df.dx).array()
    mny = df.assemble(my*df.TestFunction(CG) * df.dx).array()
    mnz = df.assemble(mz*df.TestFunction(CG) * df.dx).array()
    volumes = df.assemble(df.TestFunction(CG) * df.dx).array()
    volumes3 = df.assemble(CG3_vector * df.dx).array()
    
    m_int_arr = m_int.vector().array()
    m_int_arr.shape=(3,-1)

    m_int_arr[0][:]=mnx/volumes
    m_int_arr[1][:]=mny/volumes
    m_int_arr[2][:]=mnz/volumes
    
    m_int_arr/=np.sqrt(m_int_arr[0]**2+m_int_arr[1]**2+m_int_arr[2]**2)

    print 'length4',np.sqrt(m_int_arr[0]**2+m_int_arr[1]**2+m_int_arr[2]**2)
    
    
    m_int_arr.shape=(-1,)
    
    print m_cg.vector().array()-m_int_arr
    
    
    mn = df.assemble(df.dot(m_dg, df.Constant([1, 1, 1])) * CG3_vector * df.dx).array()/volumes3
    mn.shape=(3,-1)
    mn/=np.sqrt(mn[0]**2+mn[1]**2+mn[2]**2)
    mn.shape=(-1,)
    print 'diff3',m_cg.vector().array()-mn
    
    m_cg.vector().set_local(m_int_arr)
     
    #df.plot(m_int)
    df.plot(m_cg)
    df.interactive()
    
    


