import numpy as np
import sys
import traceback

import demo3_module

a = np.array([[1, 2], [3, 4]], dtype=float)

print "Providing a vector instead of a matrix as an argument\n"
try:
    demo3_module.trace(a[0])
    raise Exception("An exception has not been raised")
except:
    traceback.print_exception(*sys.exc_info())

print "------------------\nProviding a non-contiguous array\n"
try:
    demo3_module.trace(a.T)
    raise Exception("An exception has not been raised")
except:
    traceback.print_exception(*sys.exc_info())
