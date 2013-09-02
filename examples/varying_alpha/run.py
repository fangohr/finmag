import dolfin as df
from finmag.sim.llg import LLG

x0 = 0; x1 = 100e-9; xn = 50;
y0 = 0; y1 = 10e-9; yn = 5;
nanowire = df.RectangleMesh(x0, y0, x1, y1, xn, yn, "left/right")
S1 = df.FunctionSpace(nanowire, "Lagrange", 1)
S3 = df.VectorFunctionSpace(nanowire, "Lagrange", 1, dim=3)

llg = LLG(S1, S3)

"""
We want to increase the damping at the boundary of the object.
It is convenient to channel the power of dolfin expressions for this task.

"""

alpha_expression = df.Expression("(x[0] > x_limit) ? 1.0 : 0.5", x_limit=80e-9)
llg.set_alpha(alpha_expression)

print "alpha vector:\n", llg.alpha.vector().array()
df.plot(llg.alpha, interactive=True)
