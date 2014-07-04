#!/usr/bin/env python
import os

#$ python setup.py build_ext --inplace



from numpy.distutils.command import build_src

import Cython.Compiler.Main
build_src.Pyrex = Cython
build_src.have_pyrex = False
def have_pyrex():
    import sys
    try:
        import Cython.Compiler.Main
        sys.modules['Pyrex'] = Cython
        sys.modules['Pyrex.Compiler'] = Cython.Compiler
        sys.modules['Pyrex.Compiler.Main'] = Cython.Compiler.Main
        return True
    except ImportError:
        return False
build_src.have_pyrex = have_pyrex

def configuration(parent_package='',top_path=None):
    INCLUDE_DIRS = ['/usr/lib/openmpi/include/']
    LIBRARY_DIRS = []
    LIBRARIES    = []

    # PETSc
    PETSC_DIR  = os.environ.get('PETSC_DIR','/usr/lib/petsc')
    PETSC_ARCH = os.environ.get('PETSC_ARCH', 'linux-gnu-c-opt')
        
    if os.path.isdir(os.path.join(PETSC_DIR, PETSC_ARCH)):
        INCLUDE_DIRS += [os.path.join(PETSC_DIR, 'include')]
        LIBRARY_DIRS += [os.path.join(PETSC_DIR, PETSC_ARCH, 'lib')]
    else:
        raise Exception('Seems PETSC_DIR or PETSC_ARCH are wrong!')
    LIBRARIES += [#'petscts', 'petscsnes', 'petscksp',
                  #'petscdm', 'petscmat',  'petscvec',
                  'petsc']

    # PETSc for Python
    import petsc4py
    INCLUDE_DIRS += [petsc4py.get_include()]
    
    print "[DDD] INCLUDE_DIRS = {}".format(INCLUDE_DIRS)
    
    # Configuration
    from numpy.distutils.misc_util import Configuration
    config = Configuration('', parent_package, top_path)

    
    config.add_extension('clib',
                         sources = ['clib.pyx', 'util.c'],
                         depends = [''],
                         include_dirs=INCLUDE_DIRS + [os.curdir],
                         libraries=LIBRARIES,
                         library_dirs=LIBRARY_DIRS,
                         runtime_library_dirs=LIBRARY_DIRS)
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
