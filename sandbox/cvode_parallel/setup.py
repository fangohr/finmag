from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
import petsc4py

#python setup.py build_ext --inplace

import fnmatch
import os
import glob

#print __file__
#print os.getcwd()
realpath=os.path.realpath(__file__)
par_path=os.path.split(realpath)[0]

sundials_path = os.path.join(par_path,'sundials')

sources = []
sources.append(os.path.join(sundials_path,'cvode2.pyx'))


ext_modules = [
    Extension("cvode2",
              sources = sources,
              include_dirs = [numpy.get_include(),petsc4py.get_include(),'/usr/include/petsc', '/usr/include/mpi'],
              libraries=['m','sundials_cvodes','sundials_nvecserial'],
              extra_compile_args=["-fopenmp"],
              extra_link_args=['-fopenmp'],
              #extra_link_args=["-g"],
        )
    ]
    

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
