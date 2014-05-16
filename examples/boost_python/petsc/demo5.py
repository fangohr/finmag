import dolfin as df
import demo5_module

mesh = df.UnitCubeMesh(5, 5, 5)
print "Number of vertices:", demo5_module.get_num_vertices(mesh)

V = df.FunctionSpace(mesh, 'Lagrange', 1)

expr = df.Expression('sin(x[0])')
M = df.interpolate(expr, V)

print 'vector length',demo5_module.get_vector_local_size(M.vector())
