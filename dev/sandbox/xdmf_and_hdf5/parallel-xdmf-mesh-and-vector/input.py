
from dolfin import *

mesh2 = Mesh("mesh.xdmf")
print("Mesh we have read: {}".format(mesh2.coordinates().shape))

print("Can't read back from xdmf file, see\nhttps://answers.launchpad.net/dolfin/+question/222230 ")

#V2 = FunctionSpace(mesh2, 'CG', 1)
#u2 = Function(V2, 'u.xdmf')
#u << f
#print "vector we have read: {}".format(u2.vector().array().shape)
#del f2
