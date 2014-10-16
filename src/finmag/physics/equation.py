"""
This module solves the LLG equation or one of its many variants.

Using native code tied in by instant, it allows to specify parameters
and the terms of the equation that are to be used and then solves for dm/dt.

No effective field computation, no saving of magnetisation to file or
whatever, just straight up solving of the equation of motion.

"""
import logging
import instant
import dolfin as df
from os import path

log = logging.getLogger(name="finmag")


# TODO: use field class objects instead of dolfin vectors
def Equation(m, H, dmdt):
    """
    Returns equation object initialised with dolfin vectors m, H and dmdt.

    """
    equation_module = get_equation_module(True)
    return equation_module.Equation(m, H, dmdt)


def get_equation_module(for_distribution=False):
    """
    Returns extension module that deals with the equation of motion.
    Will try to return from cache before recompiling.

    By default, dolfin will chose a cache directory using a digest of our code
    and some version numbers. This procedure enables dolfin to detect changes
    to our code and recompile on the fly. However, when we distribute FinMag we
    don't need or want on the fly recompilation and we'd rather have the
    resulting files placed in a directory known ahead of time. For this, call
    this function once with `for_distribution` set to True and ship FinMag
    including the directory build/equation.

    During normal use, our known cache directory is always checked before
    dolfin's temporary ones. Its existence bypasses on the fly recompilation.

    """
    # __file__ will not be available during module init if this module is
    # compiled with cython. So the following line shouldn't be moved to the
    # module level. It is perfectly safe inside this function though.
    MODULE_DIR = path.dirname(path.abspath(__file__))
    SOURCE_DIR = path.join(MODULE_DIR, "native")
    # Define our own cache base directory instead of the default one. This
    # helps in distributing only the compiled code without sources.
    CACHE_DIR = path.join(MODULE_DIR, "build")

    signature = "equation" if for_distribution else ""  # dolfin will chose

    # Try to get the module from the known distribution location before
    # asking instant about its cache. This way a distributed copy of FinMag
    # should never attempt recompilation (which would fail without sources).
    equation_module = instant.import_module("equation", CACHE_DIR)
    if equation_module is not None:
        log.debug("Got equation extension module from distribution location.")
    else:
        with open(path.join(SOURCE_DIR, "equation.h"), "r") as header:
            code = header.read()

        equation_module = df.compile_extension_module(
            code=code,
            sources=["equation.cpp", "terms.cpp", "derivatives.cpp"],
            source_directory=SOURCE_DIR,  # where the sources given above are
            include_dirs=[SOURCE_DIR],  # where to look for header files
            # dolfin's compile_extension_module will pass on `module_name` to
            # instant's build_module as `signature`. That's the name of the
            # directory it will be cached in. So don't worry if instant's doc
            # says that passing a module name will disable caching.
            module_name=signature,
            cache_dir=CACHE_DIR,)

    return equation_module
