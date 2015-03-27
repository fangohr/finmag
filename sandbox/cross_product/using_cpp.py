import dolfin as df
from finmag.physics.equation import Equation

mesh = df.IntervalMesh(10, 0, 1)
S1 = df.FunctionSpace(mesh, "CG", 1)
S3 = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)

alpha = df.Function(S1)
m = df.Function(S3)
H = df.Function(S3)
dmdt = df.Function(S3)

alpha.assign(df.Constant(1))
m.assign(df.Constant((1, 0, 0)))
H.assign(df.Constant((0, 1, 0)))

eq = Equation(m.vector(), H.vector(), dmdt.vector())
eq.set_alpha(alpha.vector())
eq.set_gamma(1.0)
eq.solve()

print dmdt.vector().array()
