from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os

#python setup.py build_ext --inplace
#NFFT_DIR = os.path.expanduser('~/nfft-3.2.0')


ext_modules = [
    Extension("fast_sum_lib",
              sources = ['fast_sum.c','fast_sum_lib.pyx'],
              include_dirs = [numpy.get_include()],
              libraries=['m'],
              #extra_compile_args=["-fopenmp"],
              #extra_link_args=["-g"],
        )
    ]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)

