print "Hello World"
import dolfin as df
nx = ny = 10
mesh = df.RectangleMesh(0, 0, 1, 1, nx, ny)
V = df.FunctionSpace(mesh, 'CG', 1)
zero = df.Constant(0.)
zero_v = df.interpolate(zero, V)
print("My vector is of shape %s." % zero_v.vector().array().shape)
