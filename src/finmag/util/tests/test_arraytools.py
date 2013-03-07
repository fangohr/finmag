import numpy
from finmag.util.arraytools import flat, unflat, vecmag3d

import pytest

def test_flat():
    #
    a=numpy.arange(15)# think of 5 vectors with 3 components
    b=numpy.ndarray.view(a)
    b.shape=(3,-1)

    #check flattening works
    assert (flat(b)==a).all()
    
    #check flat is a view
    c=flat(b)
    b[0,0]=42
    assert c[0]==42

def test_unflat():
    #
    a=numpy.arange(15)# think of 5 vectors with 3 components
    b=numpy.ndarray.view(a)
    b.shape=(3,-1)

    #check unflattening works
    assert (unflat(a)==b).all()
    
    #check flat is a view
    c=unflat(a)
    c[0,0]=42
    assert a[0]==42

    #check that only data is accepted that is compatible with 3d
    a=numpy.arange(10)# think of 5 vectors with 2 components
    b=numpy.ndarray.view(a)
    b.shape=(2,-1)
    pytest.raises(AssertionError,lambda : unflat(a))

def test_magnitude():
    a=numpy.array([[0,0,1],[0,2,0],[3,0,0],[3,4,0]]).ravel('f')
    assert  (vecmag3d(unflat(a))==numpy.array([1,2,3,5])).all()

def test_dolfinvectordatalayout():
    """Check that the dolfin data layout is compatible with the flat/unflat
    functions."""
    import dolfin
    mesh=dolfin.IntervalMesh(10,0,1)
    V = dolfin.VectorFunctionSpace(mesh,"CG",1,dim=3)
    f = dolfin.Expression(("0.","1.0","2.0"))
    u = dolfin.interpolate(f,V)
    data = u.vector().array()
    print unflat(data)
    assert (unflat(data)[0,:] == 0.).all() #check x-component
    assert (unflat(data)[1,:] == 1.).all() #y-component
    assert (unflat(data)[2,:] == 2.).all() #z-component


if __name__=="__main__":
    test_flat()
    test_unflat()
    test_magnitude()



##aim: calulate the length of every 3d-column vector in b:
#
#print "sum of squares for every vector"
#print numpy.sum(b**2,-2)
#
#print "check whether this is smaller than c"
#c=20
#numpy.less(numpy.sum(b**2,-2),c**2)
#
#print "Assert this for all entries"
#
#assert numpy.less(numpy.sum(b**2,-2),c**2).all()
#
#print "Similar to check to see whether vector nomr is greater than b"
#d=10
#
#assert numpy.greater(numpy.sum(b**2,-2),d**2).all()

