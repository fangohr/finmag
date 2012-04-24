import dolfin as df
from finmag.sim.llg import LLG

x0 = 0; x1 = 100e-9; xn = 50;
y0 = 0; y1 = 10e-9; yn = 5;
nanowire = df.Rectangle(x0, y0, x1, y1, xn, yn, "left/right")

llg = LLG(nanowire)

"""
We want to increase the damping at the boundary of the object. While it would
be possible to provide a vector of values for alpha directly with llg.alpha_vec,
it is more convenient to channel the power of dolfin expressions for this
task.

"""

mult = df.Function(llg.F)
mult.assign(df.Expression("(x[0]>x_limit) ? 2.0 : 1.0", x_limit=80e-9))
llg.spatially_varying_alpha(0.5, mult)

print "baseline alpha: ", llg.alpha
print "alpha vector:\n", llg.alpha_vec
