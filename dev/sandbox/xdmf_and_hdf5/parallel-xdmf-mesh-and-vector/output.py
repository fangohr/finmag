
from dolfin import *

mesh = UnitSquareMesh(5, 5)
V = FunctionSpace(mesh, 'CG', 1)
u = interpolate(Expression('x[0]'), V)

f = File('mesh.xdmf')

f << mesh
print("mesh we have written: {}".format(mesh.coordinates().shape))
del f


fu = File('u.xdmf')
fu << u
print("u we have written: {}".format(u.vector().array().shape))

del fu


