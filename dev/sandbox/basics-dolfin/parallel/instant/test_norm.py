from dolfin import *
import numpy as np
import os

def test_norm():
    header_file = open("Foo/Foo.h", "r")
    code = header_file.read()
    header_file.close()
    foo_module = compile_extension_module(
        code=code, source_directory="Foo", sources=["Foo.cpp", "Bar.cpp"],
        include_dirs=[".", os.path.abspath("Foo")])

    mesh = UnitIntervalMesh(2)

    V = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
    field = Function(V)
    field.assign(Expression(("1", "2", "3")))
    print "field", field.vector().array()
    print "shape of field.vector.array", field.vector().array().shape

    W = FunctionSpace(mesh, 'CG', 1)
    norm = Function(W)
    print "norm", norm.vector().array()
    print "shape of norm.vector.array", norm.vector().array().shape

    foo = foo_module.Foo()
    foo.norm(field.vector(), norm.vector())

    print "norm after computation", norm.vector().array()
    expected = np.zeros(mesh.num_vertices())
    expected += np.sqrt(1 + 2*2 + 3*3)
    assert np.allclose(norm.vector().array(), expected)

