# Build script for the treecode/PBC BEM Cython extension.
#
# Historically this used the (now removed on Python 3.12) ``distutils`` and
# ``Cython.Distutils.build_ext``. It is ported to the setuptools + Cython
# ``cythonize`` build so it compiles unchanged on the DOLFINx pixi lane
# (Python 3.12, NumPy 2.x). The C kernels (``common.c``, ``bem_pbc.c``,
# ``treecode_bem_I.c``, ``treecode_bem_II.c``) are pure C + the NumPy C-API
# (no dolfin), so the only build-time addition over the legacy toolchain is
# Cython itself. [Claude Opus 4.8]
#
#   python setup.py build_ext --inplace
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "treecode_bem",
        sources=[
            "common.c",
            "bem_pbc.c",
            "treecode_bem_I.c",
            "treecode_bem_II.c",
            "treecode_bem_lib.pyx",
        ],
        include_dirs=[numpy.get_include()],
        libraries=["m"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="treecode_bem",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={"language_level": "3"},
    ),
)
