print("Hello World")
import dolfin as df
#df.parameters.reorder_dofs_serial = False
nx = ny = 10
mesh = df.RectangleMesh(df.Point(0, 0), df.Point(1, 1), nx, ny)
V = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
vector_value_constant = df.Constant(list([0,1,2]))
our_function = df.interpolate(vector_value_constant, V)
print("My vector is of shape %s." % our_function.vector().array().shape)
print(our_function.vector().array())
