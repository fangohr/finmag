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

def test_norms():
    v = [1, 1, 0]
    assert abs(norm(v) - np.sqrt(2)) < TOLERANCE

    v = np.array([[1, 1, 0], [1, -2, 3]])
    assert np.allclose(norm(v), np.array([np.sqrt(2), np.sqrt(14)]), rtol=TOLERANCE)

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
    expected = a5 / np.sqrt(2)
    assert np.allclose(fnormalise(a5), expected, rtol=TOLERANCE)

def test_vector_valued_function():
    """
    Test that the different ways of initialising a vector-valued
    function on a 3d mesh work and that they produce the expected
    results.

    """
    mesh = df.UnitCube(2, 2, 2)
    mesh.coordinates()[:] += 1.0  # shift mesh coords to avoid dividing by zero when normalising below
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    num_vertices = mesh.num_vertices()

    vec = np.array([3, 1, 4])  # an arbitrary vector
    vec_normalised = vec/norm(vec)
    a = 42
    b = 5
    c = 23

    # Reference vector for the constant-valued functions
    x = np.empty((num_vertices, 3))
    x[:] = vec
    v_ref = x.transpose().reshape(-1)
    v_ref_normalised = fnormalise(v_ref[:])

    # Reference vector for f_expr and f_callable
    v_ref_expr = (mesh.coordinates()*[a, b, c]).transpose().reshape((-1,))
    v_ref_expr_normalised = fnormalise(v_ref_expr)

    # Create functions using the various methods
    f_tuple = vector_valued_function(tuple(vec), S3) # 3-tuple
    f_list = vector_valued_function(list(vec), S3) # 3-list
    f_array3 = vector_valued_function(np.array(vec), S3) # numpy array representing a 3-vector
    f_dfconstant = vector_valued_function(df.Constant(vec), S3) # df. Constant representing a 3-vector
    f_expr = vector_valued_function(('a*x[0]', 'b*x[1]', 'c*x[2]'), S3, a=a, b=b, c=c) # tuple of strings (will be cast to df.Expression)
    f_arrayN = vector_valued_function(v_ref, S3) # numpy array of nodal values
    f_callable = vector_valued_function(lambda coords: v_ref_expr, S3) # callable accepting mesh node coordinates and yielding the function values

    # A few normalised versions, too
    f_tuple_normalised = vector_valued_function(tuple(vec), S3, normalise=True)
    f_expr_normalised = vector_valued_function(('a*x[0]', 'b*x[1]', 'c*x[2]'), S3, a=a, b=b, c=c, normalise=True)
    f_callable_normalised = vector_valued_function(lambda coords: v_ref_expr, S3, normalise=True)

    # Check that the function vectors are as expected
    assert(all(f_tuple.vector() == v_ref))
    assert(all(f_list.vector() == v_ref))
    assert(all(f_array3.vector() == v_ref))
    assert(all(f_dfconstant.vector() == v_ref))
    assert(all(f_expr.vector() == v_ref_expr))
    assert(all(f_arrayN.vector() == v_ref))
    assert(all(f_callable.vector() == v_ref_expr))

    assert(all(f_tuple_normalised.vector() == v_ref_normalised))
    print "[DDD] #1: {}".format(f_expr_normalised.vector().array())
    print "[DDD] #2: {}".format(v_ref_expr_normalised)

    assert(all(f_expr_normalised.vector() == v_ref_expr_normalised))
    assert(all(f_callable_normalised.vector() == v_ref_expr_normalised))

def test_angle():
    assert abs(angle([1,0,0],[1,0,0]))           < TOLERANCE
    assert abs(angle([1,0,0],[0,1,0]) - np.pi/2) < TOLERANCE
    assert abs(angle([1,0,0],[1,1,0]) - np.pi/4) < TOLERANCE

def test_rows_to_columns():
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    assert np.array_equal(y, rows_to_columns(x))

def test_cartesian_to_spherical():
    hapi = np.pi / 2
    test_vectors = np.array((
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (-1, 0, 0), (0, -2, 0), (0, 0, -1)))
    expected = np.array((
        (1, hapi, 0), (1, hapi, hapi), (1, 0, 0),
        (1, hapi, np.pi), (2, hapi, -hapi), (1, np.pi, 0)))
    for i, v in enumerate(test_vectors):
        v_spherical = cartesian_to_spherical(v)
        print "Testing vector {}. Got {}. Expected {}.".format(v, v_spherical, expected[i])
        assert np.max(np.abs(v_spherical - expected[i])) < TOLERANCE
