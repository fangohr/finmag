"""
This module will one day solve the LLG equation or one of its many variants.

Using native code tied in by instant, it shall allow to specify parameters
and the terms of the equation that are to be used and then solve for dm/dt.

No effective field computation, no saving of magnetisation to file or
whatever, just straight up solving of the equation.

"""
import os
import dolfin as df

with open("native/equation.h", "r") as header:
    code = header.read()

equation_module = df.compile_extension_module(
        code=code,
        source_directory="native",
        sources=["equation.cpp", "terms.cpp"],
        include_dirs=[".", os.path.abspath("native")],)
