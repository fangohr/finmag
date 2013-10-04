
import dolfin as df
mesh = df.BoxMesh(0,0,0,1,1,1, 10, 10, 10)

V1 = df.VectorFunctionSpace(mesh,"CG",1)
VT = df.TensorFunctionSpace(mesh,"CG",1)
Vs = df.FunctionSpace(mesh,"CG",1)
tf = df.TestFunction(Vs)



#from finmag.energies.dmi import dmi_term3d, dmi_term2d, dmi_term3d_dolfin

def compare_dmi_term3d_with_dolfin(Mexp):
    """Expects string to feed into df.Expression for M"""
    print "Working on Mexp=",Mexp
    Mexp=df.Expression(Mexp)
    M = df.interpolate(Mexp,V1)
    E = dmi_term3d(M,tf,1)[0] * df.dx
    E1=df.assemble(E)
    E_dolfin=dmi_term3d_dolfin(M,tf,1)[0] * df.dx
    dolfin_curl = df.project(df.curl(M),V1)
    curlx,curly,curlz = dolfin_curl.split()
    print "dolfin-curlx=",df.assemble(curlx*df.dx)
    print "dolfin-curly=",df.assemble(curly*df.dx)
    print "dolfin-curlz=",df.assemble(curlz*df.dx)
    E2=df.assemble(E_dolfin)
    print E1,E2
    print "Diff is %.18e" % (E1-E2)
    return abs(E1-E2)




def compare_dmi_term2d_with_dolfin(Mexp):
    """Expects string to feed into df.Expression for M"""
    print "Working on Mexp=",Mexp
    Mexp=df.Expression(Mexp)
    V2d = df.VectorFunctionSpace(mesh,"CG",1)
    M2d = df.interpolate(Mexp,V2d)
    M = df.interpolate(Mexp,V1)
    E = dmi_term2d(M2d,tf,1)[0] * df.dx
    E1=df.assemble(E)
    E_dolfin=dmi_term3d_dolfin(M,tf,1)[0] * df.dx
    dolfin_curl = df.project(df.curl(M),V1)
    curlx,curly,curlz = dolfin_curl.split()
    print "dolfin-curlx=",df.assemble(curlx*df.dx)
    print "dolfin-curly=",df.assemble(curly*df.dx)
    print "dolfin-curlz=",df.assemble(curlz*df.dx)
    E2=df.assemble(E_dolfin)
    print E1,E2
    print "Diff is %.18e" % (E1-E2)
    return abs(E1-E2)

def dis_test_dmi_term2d():
    mesh = df.BoxMesh(0,0,0,1,1,1, 10, 10, 10)
    mesh2d = df.RectangleMesh(0,0,1,1,10,10)
    
    
    eps=1e-15
    assert compare_dmi_term2d_with_dolfin(("x[0]","0.","0."))<eps 
    assert compare_dmi_term2d_with_dolfin(("x[1]","0.","0."))<eps
    assert compare_dmi_term2d_with_dolfin(("x[2]","0.","0."))<eps
    assert compare_dmi_term2d_with_dolfin(("0","x[0]","0.")) <eps
    assert compare_dmi_term2d_with_dolfin(("0","x[1]","0.")) <eps
    assert compare_dmi_term2d_with_dolfin(("0","x[2]","0.")) <eps
    #assert compare_dmi_term2d_with_dolfin(("0.","0","x[0]")) <eps
    #assert compare_dmi_term2d_with_dolfin(("0.","0","x[1]")) <eps
    #assert compare_dmi_term2d_with_dolfin(("0.","0","x[2]")) <eps

    #and some more complicated expressions
    assert compare_dmi_term2d_with_dolfin(("-0.5*x[1]","0.5*x[0]","1"))<eps
    assert compare_dmi_term2d_with_dolfin(("-0.5*x[1]*x[1]",
                                           "2*0.5*x[0]",
                                           "0"))<eps
    assert compare_dmi_term2d_with_dolfin(("-0.5*x[1]*x[0]",
                                           "2*0.5*x[0]-x[1]",
                                           "0"))<eps



def dis_test_dmi_with_analytical_solution():
    """For a vector field a(x,y,z)=0.5 * (-y, x, c), 
    the curl is exactly 1.0."""
    
    eps=1e-13
    M = df.interpolate(df.Expression(("-0.5*x[1]","0.5*x[0]","1")),V1)
    c=1.0
    E1 = df.assemble(dmi_term3d(M,tf,c)[0] * df.dx)
    Eexp = 1.0
    print "Expect E=%e, computed E=%e" % (Eexp,E1)
    diff = abs(E1-Eexp)
    print "deviation between analytical result and numerical is %e" % diff
    assert diff<eps

    """For a vector field a(x,y,z)=0.5 * (-y, x, c), 
    the curl is exactly 1.0."""
    eps=1e-12
    M = df.interpolate(df.Expression(("-0.5*x[1]*2","0.5*x[0]*2","1")),V1)
    c=3.0
    E1 = df.assemble(dmi_term3d(M,tf,c)[0] * df.dx)
    Eexp = 6.0
    print "Expect E=%e, computed E=%e" % (Eexp,E1)
    diff = abs(E1-Eexp)
    print "deviation between analytical result and numerical is %e" % diff
    assert diff<eps



def dis_test_dmi_term3d():
    eps=1e-15
    assert compare_dmi_term3d_with_dolfin(("x[0]","0.","0."))<eps 
    assert compare_dmi_term3d_with_dolfin(("x[1]","0.","0."))<eps
    assert compare_dmi_term3d_with_dolfin(("x[2]","0.","0."))<eps
    assert compare_dmi_term3d_with_dolfin(("0","x[0]","0.")) <eps
    assert compare_dmi_term3d_with_dolfin(("0","x[1]","0.")) <eps
    assert compare_dmi_term3d_with_dolfin(("0","x[2]","0.")) <eps
    assert compare_dmi_term3d_with_dolfin(("0.","0","x[0]")) <eps
    assert compare_dmi_term3d_with_dolfin(("0.","0","x[1]")) <eps
    assert compare_dmi_term3d_with_dolfin(("0.","0","x[2]")) <eps

    #and some more complicated expressions
    assert compare_dmi_term3d_with_dolfin(("-0.5*x[1]","0.5*x[0]","1"))<eps
    assert compare_dmi_term3d_with_dolfin(("-0.5*x[1]*x[1]",
                                           "2*0.5*x[0]",
                                           "x[0]+x[1]+x[2]"))<eps
    assert compare_dmi_term3d_with_dolfin(("-0.5*x[1]*x[0]",
                                           "2*0.5*x[0]-x[2]",
                                           "x[0]+x[1]+x[2]"))<eps


def dis_test_can_post_process_form():
    M = df.interpolate(df.Expression(("-0.5*x[1]","0.5*x[0]","1")),V1)
    c=1.0
    E = dmi_term3d(M,tf,c)[0] * df.dx

    v = df.TestFunction(V1)
    dE_dM = df.derivative(E, M, v)
    #vol = df.assemble(df.dot(v, df.Constant([1,1,1]))*df.dx).array()
    tmp = df.assemble(dE_dM)

    g_form = df.derivative(dE_dM, M)

    g_petsc = df.PETScMatrix()
        
    df.assemble(g_form,tensor=g_petsc)
    #H_dmi_petsc = df.PETScVector()

    #if we got to this line, the required assembly to compute fields works.
    assert True 


if __name__=="__main__":
    #test_dmi_term3d()
    #test_dmi_term2d()
    #test_can_post_process_form()
    test_dmi_with_analytical_solution()
