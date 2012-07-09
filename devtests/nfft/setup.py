from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


NFFT_DIR = "/Users/ww1g11/Softwares/nfft-3.2.0"

ext_modules = [
    Extension("demag_nfft_lib",
              sources = ['demag_nfft.c','demag_nfft_lib.pyx'],
              include_dirs = ['%s/include'%NFFT_DIR,numpy.get_include()],
              libraries=['m','fftw3','nfft3']
        )
    ]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
