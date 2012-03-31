"""
Some tools to convert dolfin flat numpy arrays of shape (3n,) into
numpy arrays of shape (3,n). Expect to need this repeatedly.
"""
import numpy

def unflat(a):
    """returns a view of the vector a with 3 rows, and len(a)/3 columns"""
    assert len(a)//3 == len(a)/3.
    newview = numpy.ndarray.view(a)
    newview.shape = (3,-1)
    return newview

def flat(a):
    """returns a flat view of the array"""
    return a.ravel()

def vecmag3dsquared(a):
    assert a.shape[0]==3,"expect a matrix with shape (3,n)"
    return numpy.sum(a**2,-2)

def vecmag3d(a):
    assert a.shape[0]==3,"expect a matrix with shape (3,n)"
    return numpy.sqrt(vecmag3dsquared(a))

