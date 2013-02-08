# Basics taken from 
# --------- Forwarded message ----------
# From: Johan Hake <hake.dev@gmail.com>
# Date: 2013/1/28
# Subject: [Dolfin] DofMap.tabulate_vertex_map
# To: dolfin-dev <dolfin@lists.launchpad.net>
#


import dolfin
ver = dolfin.__version__
major, minor, revision = map(int, dolfin.__version__.split('.'))
assert major >= 1
assert minor >= 1, "Need dolfin >= 1.1 for this"

from dolfin import *
import numpy as np
mesh = UnitSquareMesh(10,10)
V = VectorFunctionSpace(mesh, "CG", 1)
u = Function(V)
u.interpolate(Constant((1,2)))
vert_values = np.zeros(mesh.num_vertices()*2)
vert_values[V.dofmap().tabulate_vertex_map(mesh)] \
    = u.vector().array()
print vert_values
