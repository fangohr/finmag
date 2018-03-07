from dolfin import *
import dolfin as df
from fractions import Fraction
import numpy

def my_print_array(a):
    (m,n)=a.shape
    for j in range(n):
        for i in range(m):
            x=a[i][j]
            y= Fraction(x).limit_denominator()
            print "%6s"%y,
        print ''



def compute_assemble(u1,v1,u3,v3,dolfin_str):
    tmp_str='A=df.assemble(%s)'%(dolfin_str)
    exec tmp_str
    print '='*70,dolfin_str,A.array().shape
    
    my_print_array(A.array())
    

    
    

def test_assemble(mesh):
    V = FunctionSpace(mesh, 'Lagrange', 1)
    Vv=VectorFunctionSpace(mesh, 'Lagrange', 1)

    u1 = TrialFunction(V)
    v1 = TestFunction(V)
    u3 = TrialFunction(Vv)
    v3 = TestFunction(Vv)

    
    test_list=[
        "u1*v1*dx",
        "inner(grad(u1),grad(v1))*dx",
        "inner(u3,v3)*dx",
        "inner(grad(u1), v3)*dx",
        "inner(u3, grad(v1))*dx"
        ]
    
    for i in test_list:
        compute_assemble(u1,v1,u3,v3,i)
    


if __name__ == "__main__":
        
    mesh = df.UnitCubeMesh(1,1,1)
    mesh = RectangleMesh(0, 0, 2, 2, 1, 1)
    test_assemble(mesh)

    
    



    
