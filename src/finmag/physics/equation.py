"""
This module solves the LLG equation or one of its many variants.

Using native code tied in by instant, it allows to specify parameters
and the terms of the equation that are to be used and then solves for dm/dt.

No effective field computation, no saving of magnetisation to file or
whatever, just straight up solving of the equation of motion.

"""
from os import path
import dolfin as df

MODULE_DIR = path.dirname(path.abspath(__file__))
NATIVE_DIR = path.join(MODULE_DIR, "native")

with open(path.join(NATIVE_DIR, "equation.h"), "r") as header:
    code = header.read()

equation_module = df.compile_extension_module(
    code=code,
    source_directory=NATIVE_DIR,
    sources=["equation.cpp", "terms.cpp", "derivatives.cpp"],
    include_dirs=[NATIVE_DIR],)

Equation = equation_module.Equation
