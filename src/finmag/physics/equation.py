"""
This module will one day solve the LLG equation or one of its many variants.

Using native code tied in by instant, it shall allow to specify parameters
and the terms of the equation that are to be used and then solve for dm/dt.

No effective field computation, no saving of magnetisation to file or
whatever, just straight up solving of the equation.

"""
import os

with open("native/equation.h", "r") as header:
    code = header.read()

equation_module = df.compile_extension_module(
        code=code,
        source_directory="native",
        sources=["equation.cpp"],
        include_dirs=[".", os.path.abspath("native")],)

# for testing purposes

import numpy as np
import dolfin as df

mesh = df.UnitIntervalMesh(2)
V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
m = df.Function(V)
dmdt = df.Function(V)
equation = equation_module.Equation(m.vector(), dmdt.vector())
equation.solve()
