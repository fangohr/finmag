"""
An example code that computes the cross product of two dolfin functions.
This should be replaced in the parts of the code where dmdt is computed.
"""
import dolfin as df
from distutils.version import LooseVersion

if LooseVersion(df.__version__) < LooseVersion('1.5.0'):
    raise RuntimeError("This script requires at least dolfin version 1.5. It will run without error in 1.4, but the computed cross product will be zero!")

def cross_product(a, b, S3):
    """
    Function computing the cross product of two vector functions.
    The result is a dolfin vector.
    """
    return df.assemble(df.dot(df.cross(a, b), df.TestFunction(S3)) * df.dP)

mesh = df.IntervalMesh(10, 0, 1)
S3 = df.VectorFunctionSpace(mesh, 'CG', 1, 3)

a = df.Function(S3)
b = df.Function(S3)
a.assign(df.Constant((1, 0, 0)))  # unit x vector
b.assign(df.Constant((0, 1, 0)))  # unit y vector

axb = cross_product(a, b, S3)

# An expected result is unit z vector and the type is dolfin vector.
print axb.array()
print type(axb)
