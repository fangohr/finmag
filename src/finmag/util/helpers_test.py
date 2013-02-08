import numpy as np
import dolfin as df
from finmag.util.helpers import *
from finmag.util.meshes import box
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
    mesh.coordinates()[:] = mesh.coordinates() + 1.0  # shift mesh coords to avoid dividing by zero when normalising below
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    num_vertices = mesh.num_vertices()

    vec = np.array([3, 1, 4])  # an arbitrary vector
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
    f_array3xN = vector_valued_function(v_ref, S3) # numpy array of nodal values shape (3*n,)
    f_arrayN3 = vector_valued_function(np.array([vec for r in mesh.coordinates()]), S3) # numpy array of shape (n, 3)
    #f_callable = vector_valued_function(lambda coords: v_ref_expr, S3) # callable accepting mesh node coordinates and yielding the function values
    # Cython 0.17.1 does not like this
    #f_callable = vector_valued_function(lambda (x,y,z): (a*x, b*y, c*z), S3) # callable accepting mesh node coordinates and yielding the function values
    # but this one is okay
    f_callable = vector_valued_function(lambda t: (a * t[0], b * t[1], c * t[2]), S3) # callable accepting mesh node coordinates and yielding the function values

    # A few normalised versions, too
    f_tuple_normalised = vector_valued_function(tuple(vec), S3, normalise=True)
    f_expr_normalised = vector_valued_function(('a*x[0]', 'b*x[1]', 'c*x[2]'), S3, a=a, b=b, c=c, normalise=True)
    
    # Cython 0.17.1 does not like this
    #f_callable_normalised = vector_valued_function(lambda (x,y,z): (a*x, b*y, c*z), S3, normalise=True)
    # but accepts this rephrased version:
    f_callable_normalised = vector_valued_function(lambda t: (a*t[0], b*t[1], c*t[2]), S3, normalise=True)


    # Check that the function vectors are as expected
    assert(all(f_tuple.vector() == v_ref))
    assert(all(f_list.vector() == v_ref))
    assert(all(f_array3.vector() == v_ref))
    assert(all(f_dfconstant.vector() == v_ref))
    assert(all(f_expr.vector() == v_ref_expr))
    assert(all(f_array3xN.vector() == v_ref))
    assert(all(f_arrayN3.vector() == v_ref))
    assert(all(f_callable.vector() == v_ref_expr))

    assert(all(f_tuple_normalised.vector() == v_ref_normalised))
    print "[DDD] #1: {}".format(f_expr_normalised.vector().array())
    print "[DDD] #2: {}".format(v_ref_expr_normalised)

    assert(all(f_expr_normalised.vector() == v_ref_expr_normalised))
    assert(all(f_callable_normalised.vector() == v_ref_expr_normalised))
    
    
def test_scalar_valued_dg_function():
    mesh = df.UnitCube(2, 2, 2)
    
    def init_f(coord):
        x,y,z=coord
        if z<=0.5:
            return 1
        else:
            return 10
    
    f=scalar_valued_dg_function(init_f, mesh)
    
    assert f(0,0,0.51)==10.0
    assert f(0.5,0.7,0.51)==10.0
    assert f(0.4,0.3,0.96)==10.0
    assert f(0,0,0.49)==1.0
    fa=f.vector().array().reshape(2,-1)
    
    assert np.min(fa[0])==np.max(fa[0])==1
    assert np.min(fa[1])==np.max(fa[1])==10
    
    
    dg = df.FunctionSpace(mesh, "DG", 0)
    dgf=df.Function(dg)
    dgf.vector()[0]=9.9
    f=scalar_valued_dg_function(dgf, mesh)
    assert f.vector().array()[0]==9.9
    

def test_angle():
    assert abs(angle([1,0,0], [1,0,0]))           < TOLERANCE
    assert abs(angle([1,0,0], [0,1,0]) - np.pi/2) < TOLERANCE
    assert abs(angle([1,0,0], [1,1,0]) - np.pi/4) < TOLERANCE

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

def test_pointing_upwards():
    assert pointing_upwards((0, 0, 1))
    assert pointing_upwards((0.5, 0.5, 0.8))
    assert pointing_upwards((-0.5, 0.5, 0.8))
    assert not pointing_upwards((0, 0, -1))
    assert not pointing_upwards((-0.5, 0.5, -0.8))
    assert not pointing_upwards((-0.5, 0.5, 0.4))

def test_pointing_downwards():
    assert pointing_downwards((0, 0, -1))
    assert pointing_downwards((-0.5, -0.5, -0.8))
    assert pointing_downwards((-0.5, 0.5, -0.8))
    assert not pointing_downwards((0, 0, 1))
    assert not pointing_downwards((-0.5, -0.5, 0.8))
    assert not pointing_downwards((-0.5, 0.5, -0.4))


def test_mesh_functions_allclose():
    """
    First we define a cuboid mesh stretching from -2.0 to 2.0 in all
    three dimensions. Then wedefine two functions on this mesh:

        f1(r) = 1/r   where r is the distance to the origin
        f2(r) = 1.0   constant over the mesh

    Finally, we compare these two functios with absolute tolerance 1.0,
    first on the whole mesh and then only outside the unit sphere.

    The first comparison should return False because f1 takes on very
    large values close to the origin. However, the second comparison
    should return True since f1 only takes values between 0 and 1
    outside the unit sphere.
    """
    mesh = box(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0, maxh=0.5)
    CG1 = df.FunctionSpace(mesh, 'CG', 1)
    e1 = df.Expression('1.0/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])')
    f1 = df.interpolate(e1, CG1)
    e2 = df.Expression('1.0')

    def is_inside_unit_sphere(pt):
        (x, y, z) = pt
        return (x*x + y*y + z*z) < 1.0

    f2 = df.interpolate(e2, CG1)
    assert mesh_functions_allclose(f1, f2, atol=1.0) == False
    assert mesh_functions_allclose(f1, f2, fun_mask=is_inside_unit_sphere, atol=1.0) == True


def test_piecewise_on_subdomains():
    """
    Define a simple cubic mesh with three subdomains, create a function
    which takes different values on these subdomains and check that the
    resulting function really has the right values.
    """
    mesh = df.UnitCube(1, 1, 1)
    fun_vals = (42, 23, -3.14)
    g = df.MeshFunction('uint', mesh, 3)
    g.array()[:] = [1, 1, 2, 3, 1, 3]
    p = piecewise_on_subdomains(mesh, g, fun_vals)
    assert(isinstance(p, df.Function))  # check that p is a proper Function, not a MeshFunction
    assert(np.allclose(p.vector().array(), np.array([42, 42, 23, -3.14, 42, -3.14])))


def test_vector_field_from_dolfin_function():
    """
    Create a dolfin.Function representing a vector field on a mesh and
    convert it to a vector field on a regular grid using
    `vector_field_from_dolfin_function()`. Then compare the resulting
    values with the ones obtained by directly computing the field
    values from the grid coordinates and check that they coincide.
    """

    (xmin, xmax) = (-2, 3)
    (ymin, ymax) = (-1, 2.5)
    (zmin, zmax) = (0.3, 5)
    (nx, ny, nz) = (10, 10, 10)

    # Create dolfin.Function representing the vector field. Note that
    # we use linear expressions so that they can be accurately
    # represented by the linear interpolation on the mesh.
    mesh = box(xmin, ymin, zmin, xmax, ymax, zmax, maxh=1.0)
    V = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
    e = df.Expression(('-1.0 - 3*x[0] + x[1]',
                       '+1.0 + 4*x[1] - x[2]',
                       '0.3 - 0.8*x[0] - 5*x[1] + 0.2*x[2]'))
    f = df.interpolate(e, V)

    X, Y, Z = np.mgrid[xmin:xmax:nx*1j, ymin:ymax:ny*1j, zmin:zmax:nz*1j]

    # Evaluate the vector field on the grid to create the reference arrays.
    U = -1.0 - 3*X + Y
    V = +1.0 + 4*Y - Z
    W = 0.3 - 0.8*X - 5*Y + 0.2*Z

    # Now convert the dolfin.Function to a vector field and compare to
    # the reference arrays.
    X2, Y2, Z2, U2, V2, W2 = \
        vector_field_from_dolfin_function(f, (xmin, xmax), (ymin, ymax),
                                          (zmin, zmax), nx=nx, ny=ny, nz=nz)

    assert(np.allclose(X, X2))
    assert(np.allclose(Y, Y2))
    assert(np.allclose(Z, Z2))

    assert(np.allclose(U, U2))
    assert(np.allclose(V, V2))
    assert(np.allclose(W, W2))
    
if __name__ == '__main__':
    test_scalar_valued_dg_function()
