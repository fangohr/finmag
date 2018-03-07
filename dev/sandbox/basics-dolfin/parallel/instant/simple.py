import dolfin as df

mesh = df.UnitIntervalMesh(1)
V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
f = df.Function(V)
f.assign(df.Expression(("1", "2", "3")))

print f.vector().array()
