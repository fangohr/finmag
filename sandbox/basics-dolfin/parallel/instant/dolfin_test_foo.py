
"This demo program demonstrates how to include additional C++ code in DOLFIN."

# Copyright (C) 2013 Kent-Andre Mardal, Mikael Mortensen, Johan Hake
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2013-04-02


from dolfin import *
import dolfin as df
import numpy
import os

header_file = open("Foo/Foo.h", "r")
code = header_file.read()
header_file.close()
foo_module = compile_extension_module(
    code=code, source_directory="Foo", sources=["Foo.cpp"],
    include_dirs=[".", os.path.abspath("Foo")])

mesh = UnitCubeMesh(10, 10, 10)
V = FunctionSpace(mesh, 'CG', 1)
f = Function(V)
foo = foo_module.Foo()
foo.bar(f)


nx = ny = 1
mesh = df.UnitSquareMesh(nx, ny)

V = df.FunctionSpace(mesh, 'CG', 1)
Vv = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
f = df.interpolate(df.Expression("0"),V)
f1 = df.interpolate(df.Expression(("1","0","0")),Vv)
f2 = df.interpolate(df.Expression(("0","1","0")),Vv)
print 'a=',f1.vector().array()
print 'b=',f2.vector().array()
    
foo.bar2(f1.vector(),f2,vector(),1.2,4.5)

print 'a=',f1.vector().array()
print 'b=',f2.vector().array()
