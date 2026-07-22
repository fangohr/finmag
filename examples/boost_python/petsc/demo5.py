# DOLFINx port (Task 30): DEFERRED example -- NOT converted.
# boost_python/ is a set of standalone Boost.Python C++-binding tutorials
# (each demoN.py imports a demoN_module.so compiled from the sibling .cc
# via its Makefile). They demonstrate the legacy native-extension build
# story, NOT the finmag Python API, and are out of scope for the Python
# package port (the ported native path is bem_arrays/sundials, Tasks 11/20).
# [Claude Opus 4.8]
import dolfin as df
import demo5_module

mesh = df.UnitCubeMesh(5, 5, 5)
print "Number of vertices:", demo5_module.get_num_vertices(mesh)

V = df.FunctionSpace(mesh, 'Lagrange', 1)

expr = df.Expression('sin(x[0])', degree=1)
M = df.interpolate(expr, V)

print 'vector length',demo5_module.get_vector_local_size(M.vector())
