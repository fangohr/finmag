# DOLFINx port (Task 30): DEFERRED example -- NOT converted.
# boost_python/ is a set of standalone Boost.Python C++-binding tutorials
# (each demoN.py imports a demoN_module.so compiled from the sibling .cc
# via its Makefile). They demonstrate the legacy native-extension build
# story, NOT the finmag Python API, and are out of scope for the Python
# package port (the ported native path is bem_arrays/sundials, Tasks 11/20).
# [Claude Opus 4.8]
import numpy as np
import demo3_module

a = np.array([[1, 2], [3, 4]], dtype=float)
print "Trace of a:", demo3_module.trace(a)

