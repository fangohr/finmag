from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os

#python setup.py build_ext --inplace
#NFFT_DIR = os.path.expanduser('~/nfft-3.2.0')


ext_modules = [
    Extension("fastsum_lib",
              sources = ['fast_sum.c','fast_sum_lib.pyx'],
              include_dirs = [numpy.get_include()],
              libraries=['m'],
              #extra_compile_args=["-g"],
              #extra_link_args=["-g"],
        )
    ]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)


NFFT_DIR = '/usr/local'

ext_modules = [
    Extension("demag_nfft_lib",
              sources = ['demag_nfft.c','demag_nfft_lib.pyx'],
              include_dirs = ['%s/include'%NFFT_DIR,numpy.get_include()],
              #library_dirs = ['%s/lib'%NFFT_DIR],
              libraries=['m','fftw3','nfft3'],
        )
    ]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)