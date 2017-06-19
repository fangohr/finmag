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
import os
import fnmatch
import glob

from os import path

log = logging.getLogger(name="finmag")

def find_slepc():
    slepc = None
    if 'SLEPC_DIR' in os.environ:
        slepc = os.environ['SLEPC_DIR']
    else:
        # At least on Ubuntu 16.04, the header files are in
        # /usr/lib/slepcdir/3.7.2/x86_64-linux-gnu-real/include/
        # However, tried to be a bit more robust to find it.
        slepcpath = '/usr/lib/slepcdir'
        matches = []
        if os.path.isdir(slepcpath):
            for root, dirnames, filenames in os.walk(slepcpath):
                for filename in fnmatch.filter(filenames, 'slepceps.h'):
                    matches.append(root)
                    
        # Dont want fortran header files!
        matches = [match for match in matches if 'finclude' not in match]
    if matches:
        slepc = matches[0]

    if not slepc:
        raise Exception("Cannot find SLEPc header files - please set environment variable SLEPC_DIR\n"
                        "You can also modify finmag/src/physics/equation.py")

    else:
        print("Found SLEPc include files at {}".format(slepc))
        return slepc
    
        
def find_petsc():
    petsc = None
    if 'SLEPC_DIR' in os.environ:
        petsc = os.environ['PETSC_DIR']
    else:
    # At least on Ubuntu 16.04, the header files are in
    # /usr/lib/slepcdir/3.7.2/x86_64-linux-gnu-real/include/
    # However, tried to be a bit more robust to find it.
        petscpath = '/usr/lib/petscdir'
        matches = []
        if os.path.isdir(petscpath):
            for root, dirnames, filenames in os.walk(petscpath):
                for filename in fnmatch.filter(filenames, 'petscsys.h'):
                    matches.append(root)
                    # Dont want fortran header files!
        matches = [match for match in matches if 'finclude' not in match]
    if matches:
        petsc = matches[0]

    if not petsc:
        raise Exception("Cannot find PETSc header files - please set environment variable PETSC_DIR\n"
                        "You can also modify finmag/src/physics/equation.py")

    else:
        print("Found PETSc include files at {}".format(petsc))
        return petsc

# find_slepc()
# find_petsc()



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
            include_dirs=[SOURCE_DIR, find_petsc(), find_slepc()],  # where to look for header files
            # dolfin's compile_extension_module will pass on `module_name` to
            # instant's build_module as `signature`. That's the name of the
            # directory it will be cached in. So don't worry if instant's doc
            # says that passing a module name will disable caching.
            module_name=signature,
            cache_dir=CACHE_DIR,)

    return equation_module
