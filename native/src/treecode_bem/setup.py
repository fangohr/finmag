from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os

#python setup.py build_ext --inplace

ext_modules = [
    Extension("treecode_bem",
              sources = ['common.c',
                         'treecode_bem_I.c',
                         'treecode_bem_II.c',    
                         'treecode_bem_lib.pyx'],
              include_dirs = [numpy.get_include()],
	          libraries=['m'],
              #libraries=['m','gomp'],
              #extra_compile_args=["-fopenmp"],
              #extra_link_args=["-g"],
        )
    ]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)

