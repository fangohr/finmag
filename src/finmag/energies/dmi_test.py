import pytest
import numpy as np
import dolfin as df
from finmag.energies import DMI

def test_dmi_pbc2d():
    mesh = df.BoxMesh(0,0,0,1,1,0.1,5, 5, 1)
     

    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    expr = df.Expression(("0", "0", "1"))
    
    m = df.interpolate(expr, S3)
    
    dmi = DMI(1,pbc2d=True)
    dmi.setup(S3, m, 1)
    field=dmi.compute_field()
    assert np.max(field)<1e-15





if __name__ == "__main__":
   
    
    mesh = df.BoxMesh(0,0,0,1,1,0.1,5, 5, 1)
     
    S = df.FunctionSpace(mesh, "Lagrange", 1)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)

    expr = df.Expression(("0", "0", "1"))
    
    m = df.interpolate(expr, S3)
    
    dmi = DMI(1,pbc2d=True)
    dmi.setup(S3, m, 1)
    print dmi.compute_field()
    
    field=df.Function(S3)
    field.vector().set_local(dmi.compute_field())
    
    df.plot(m)
    df.plot(field)
    df.interactive()
