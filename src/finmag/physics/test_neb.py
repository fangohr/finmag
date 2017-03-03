from dolfin import *
def test_neb():
    mesh = UnitSquareMesh(8,8)

    V = FunctionSpace(mesh,"Lagrange",1)
    ME = MixedFunctionSpace([V,V,V])
    u = Function(ME)

    c, mu, n = u.split()

    z= Function(V)
    z.interpolate(Expression("sin(x[0]*x[1])"))


    print u.vector().array()

    assign(z,u.sub(0))

    print u.vector().array()
    print z.vector().array()
