import dolfin as df

mesh = df.UnitCubeMesh(5,5,5)

print "mesh = ", mesh, ", type(mesh) =", type(mesh)
print "mesh.this = ", mesh.this, ", type(mesh.this) =", type(mesh.this)
