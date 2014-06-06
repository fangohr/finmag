print "Hello World"
import dolfin as df
#df.parameters.reorder_dofs_serial = False
nx = ny = 10
mesh = df.RectangleMesh(0, 0, 1, 1, nx, ny)
V = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
zero = df.Constant(list([0,0,1]))
zero_v = df.interpolate(zero, V)
print("My vector is of shape %s." % zero_v.vector().array().shape)
print zero_v.vector().array()
