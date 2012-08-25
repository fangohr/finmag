import numpy as np
from finmag.util.helpers import *
import pytest

TOLERANCE = 1e-15

def test_components():
    x = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    y = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    assert np.array_equal(y, components(x))

def test_vectors():
    x1 = np.array([1, 1, 2, 2, 3, 3])
    y1 = np.array([[1, 2, 3], [1, 2, 3]])
    assert np.array_equal(y1, vectors(x1))

    x2 = np.array([0, 1, 1, 0, 2, 3, 3, 2, 4, 5, 5, 4])
    y2 = np.array([[0, 2, 4], [1, 3, 5], [1, 3, 5], [0, 2, 4]])
    assert np.array_equal(y2, vectors(x2))

def test_norm():
    assert norm([0, 0, 0]) == 0
    assert norm([1, 0, 0]) == 1
    assert abs(norm([1, 1, 0]) - np.sqrt(2)) < TOLERANCE
    assert abs(norm([1, 1, 1]) - np.sqrt(3)) < TOLERANCE
    assert abs(norm([-1, 0, 0]) - norm([1, 0, 0])) < TOLERANCE 
    assert abs(3*norm([1, 1, 1]) - norm(3*np.array([1, 1, 1]))) < TOLERANCE

def test_fnormalise():
    a = np.array([1., 1., 2., 2., 0., 0.])
    norm = np.sqrt(1+2**2+0**2)
    expected = a[:]/norm
    assert np.allclose(fnormalise(a), expected, rtol=TOLERANCE)

    a = np.array([1., 2., 0, 0., 1., 3.])
    n1 = np.sqrt(1+0+1)
    n2 = np.sqrt(2**2+0+3**2)
    expected = a[:]/np.array([n1,n2,n1,n2,n1,n2])
    assert np.allclose(fnormalise(a), expected, rtol=TOLERANCE)

    a = np.array([5*[1.], 5*[0], 5*[0]])
    expected = a.copy().ravel()
    assert np.allclose(fnormalise(a), expected, rtol=TOLERANCE)

    a2 = np.array([5*[2.], 5*[0], 5*[0]])
    assert np.allclose(fnormalise(a2), expected, rtol=TOLERANCE)

    #a3=np.array([0,0,3,4., 0,2,0,5, 1,0,0,0])

    #this is 0   0   3   4
    #        0   2   0   5
    #        1   0   0   0
    #
    #can also write as
    
    a3=np.array([[0,0,1.],[0,2,0],[3,0,0],[4,5,0]]).transpose()

    c=np.sqrt(4**2+5**2)
    expected = np.array([0,0,1,4/c, 0,1,0,5/c,1,0,0,0])
    print "a3=\n",a3
    print "expected=\n",expected
    print "fnormalise(a3)=\n",fnormalise(a3)
    assert np.allclose(fnormalise(a3), expected, rtol=TOLERANCE)

    #check that normalisation also works if input vector happens to be an
    #integer array
    #first with floats
    a4 = np.array([0., 1., 1.])
    c=np.sqrt(1**2+1**2) #sqrt(2)
    expected = np.array([0,1/c,1/c])
    print "a4=\n",a4
    print "expected=\n",expected
    print "fnormalise(a4)=\n",fnormalise(a4)
    assert np.allclose(fnormalise(a4), expected, rtol=TOLERANCE)

    #the same test with ints (i.e.
    a5 = np.array([0, 1, 1])
    #) will give the wrong numerical result. To avoid this, we raise 
    #an error in fnormalise.
    #
    #Maybe there are better ways of doing this, but for now we just need to
    #identify if a call with integer data takes place.
    #
    #Check that the assertion error is raised:
    with pytest.raises(AssertionError):
        fnormalise(a5)


def test_angle():
    assert abs(angle([1,0,0],[1,0,0]))           < TOLERANCE
    assert abs(angle([1,0,0],[0,1,0]) - np.pi/2) < TOLERANCE
    assert abs(angle([1,0,0],[1,1,0]) - np.pi/4) < TOLERANCE

def test_rows_to_columns():
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    assert np.array_equal(y, rows_to_columns(x))

def test_perturbed_vectors():
    quantity = 10; direction = [1, 0, 0]; length = 5
    # I could pass a fake random function to perturbed_vectors for testing,
    # but the exact behaviour of perturbed_vectors is unspecified except
    # for the number of vectors returned and their length anyways.

    vector_field = perturbed_vectors(quantity, direction, length)
    assert len(vector_field) == 10

    for v in vector_field:
        assert len(v) == 3
        assert abs(norm(v) - 5) < TOLERANCE

def test_normalize_filepath():
    assert(normalize_filepath(None, 'foo.txt') == (os.curdir, 'foo.txt'))
    assert(normalize_filepath('', 'foo.txt') == (os.curdir, 'foo.txt'))
    assert(normalize_filepath(None, 'bar/baz/foo.txt') == ('bar/baz', 'foo.txt'))
    assert(normalize_filepath(None, '/bar/baz/foo.txt') == ('/bar/baz', 'foo.txt'))
    assert(normalize_filepath('/bar/baz', 'foo.txt') == ('/bar/baz', 'foo.txt'))
    assert(normalize_filepath('/bar/baz/', 'foo.txt') == ('/bar/baz', 'foo.txt'))
    with pytest.raises(TypeError):
        normalize_filepath(42, None)
    with pytest.raises(TypeError):
        normalize_filepath(None, 23)
    with pytest.raises(ValueError):
        normalize_filepath(None, '')
    with pytest.raises(ValueError):
        normalize_filepath('bar', '')
    with pytest.raises(ValueError):
        normalize_filepath('bar', 'baz/foo.txt')
