import os
import finmag
import dolfin as df
import numpy as np


if __name__ == "__main__":
    mesh = df.BoxMesh(0, 0, 0, np.pi, 1, 1, 10, 1, 1)
    
    CG = df.FunctionSpace(mesh, "CG", 1)
    
    CG1 = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    CG2 = df.VectorFunctionSpace(mesh, "CG", 2, dim=3)
    
    
    expr = df.Expression(('sin(x[0])', 'cos(x[0])','0'))
    m_cg1 = df.interpolate(expr, CG1)
    m_cg2 = df.interpolate(expr, CG2)
    
       
    
    
    m2 = df.project(m_cg2,CG1)
    
    d2 = m2.vector().array() - m_cg1.vector().array()
    
    m3 = df.interpolate(m_cg1,CG2)

    d3 = m3.vector().array() - m_cg2.vector().array()
    
    #print d2,len(d2)
    #print d3,len(d3)
    
    #df.plot(m3)
    #df.plot(m_cg2)
    #df.interactive()

    m=m3.vector().array()
    m.shape=(3,-1)
    m/=np.sqrt(m[0]**2+m[1]**2+m[2]**2)
    m.shape=(-1)
    
    print m-m_cg2.vector().array()
    
    
    mx = df.interpolate(df.Expression('sin(x[0])'), CG)
    mnx = df.assemble(mx * df.TestFunction(CG) * df.dx).array()
    volumes = df.assemble(df.TestFunction(CG) * df.dx).array()
    #print mnx/volumes,mx.vector().array()