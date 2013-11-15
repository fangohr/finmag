import numpy as np
import dolfin as df
import tempfile
import pytest
import os
from finmag.util.helpers import *
from finmag.util.meshes import box, cylinder
from finmag.util.visualization import render_paraview_scene
from finmag.example import barmini
import finmag

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    mesh = df.UnitCubeMesh(2, 2, 2)
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
    mesh = df.UnitCubeMesh(2, 2, 2)

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
    mesh = df.UnitCubeMesh(1, 1, 1)
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


def test_probe():
    """
    Define a function on a cylindrical mesh which decays linearly in
    x-direction. Then probe this function at a number of points along
    the x-axis. This probing is done twice, once normally and once by
    supplying a function which should be applied to the probed field
    points. The results are compared with the expected values.
    """
    # Define a vector-valued function on the mesh
    mesh = cylinder(10, 1, 3)
    V = df.VectorFunctionSpace(mesh, 'Lagrange', 1, dim=3)
    f = df.interpolate(df.Expression(['x[0]', '0', '0']), V)

    # Define the probing points along the x-axis
    xs = np.linspace(-9.9, 9.9, 20)
    pts = [[x, 0, 0] for x in xs]

    def square_x_coord(pt):
        return pt[0]**2

    # Probe the field (once normally and once with an additional
    # function applied to the result). Note that the results have
    # different shapes because apply_func returns a scalar, not a
    # 3-vector.
    res1 = probe(f, pts)
    res2 = probe(f, pts, apply_func=square_x_coord)

    # Check that we get the expected results.
    res1_expected = [[x, 0, 0] for x in xs]
    res2_expected = xs**2
    assert(np.allclose(res1, res1_expected))
    assert(np.allclose(res2, res2_expected))

    # Probe at points which lie partly outside the sample to see if we
    # get masked values in the result.
    pts = [[20, 20, 0], [5, 2, 1]]
    res1 = probe(f, pts)
    res2 = probe(f, pts, apply_func=square_x_coord)
    res1_expected = np.ma.masked_array([[np.NaN, np.NaN, np.NaN],
                                        [5, 0, 0]],
                                       mask=[[True, True, True],
                                             [False, False, False]])
    res2_expected = np.ma.masked_array([np.NaN, 25], mask=[True, False])

    # Check that the arrays are masked out at the same location
    assert((np.ma.getmask(res1) == np.ma.getmask(res1_expected)).all())
    assert((np.ma.getmask(res2) == np.ma.getmask(res2_expected)).all())

    # Check that the non-masked values are the same
    assert(np.ma.allclose(res1, res1_expected))
    assert(np.ma.allclose(res2, res2_expected))


def test_get_hg_revision_info(tmpdir):
    finmag_repo = MODULE_DIR
    os.chdir(str(tmpdir))
    os.mkdir('invalid_repo')
    with pytest.raises(ValueError):
        get_hg_revision_info('nonexisting_directory')
    with pytest.raises(ValueError):
        get_hg_revision_info('invalid_repo')
    with pytest.raises(ValueError):
        get_hg_revision_info(finmag_repo, revision='invalid_revision')
    id_string = 'd330c151a7ce'
    rev_nr, rev_id, rev_date = get_hg_revision_info(finmag_repo, revision=id_string)
    assert(rev_nr == 4)
    assert(rev_id == id_string)
    assert(rev_date == '2012-02-02')


def test_binary_tarball_name(tmpdir):
    finmag_repo = MODULE_DIR
    expected_tarball_name = 'FinMag-dist__2012-02-02__rev4_d330c151a7ce_foobar.tar.bz2'
    assert(binary_tarball_name(finmag_repo, revision='d330c151a7ce', suffix='_foobar') == expected_tarball_name)


def test_plot_ndt_columns_and_plot_dynamics(tmpdir):
    """
    Simply check that we can call the command `plot_ndt_columns` with some arguments
    """
    os.chdir(str(tmpdir))
    sim = barmini()
    sim.schedule('save_ndt', every=1e-12)
    sim.run_until(1e-11)
    plot_ndt_columns('barmini.ndt', columns=['m_x', 'm_y', 'm_z', 'E_Demag', 'H_Exchange_x'],
                     outfile='barmini.png', title="Some awesome title",
                     show_legend=True, legend_loc='center', figsize=(10, 4))

    plot_dynamics('barmini.ndt', components='xz', outfile='barmini2.png', xlim=(0, 0.8e-11), ylim=(-1, 1))

    assert(os.path.exists('barmini.png'))
    assert(os.path.exists('barmini2.png'))


def test_vortex_functions():
    f_simple = vortex_simple(r=20, center=(0, 0, 1), right_handed=True)
    f_feldtkeller = vortex_feldtkeller(beta=15, center=(0, 0, 1), right_handed=False)

    funcs = [f_simple, f_feldtkeller]
    pts = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (12, 42, 23)]

    # Try calling the functions with a few points to see if we get any errors
    for f in funcs:
        for pt in pts:
            f(pt)

    # Final sanity check: f_simple should yield zero z-coordinate
    # outside the vortex core radius, and the magnetisation should
    # curl around the center.
    assert(np.allclose(f_simple((21, 0, 0)), [0, 1, 0]))
    assert(np.allclose(f_simple((-16, 16, 20)), [-1./np.sqrt(2), -1./np.sqrt(2), 0]))


def test_crossprod():
    """
    Compute the cross product of two functions f and g numerically
    using `helpers.crossprod` and compare with the analytical
    expression.

    """
    xmin = ymin = zmin = -2
    xmax = ymax = zmax = 3
    nx = ny = nz = 10
    mesh = df.BoxMesh(xmin, ymin, zmin, xmax, ymax, zmax, nx, ny, nz)
    V = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
    u = df.interpolate(df.Expression(['x[0]', 'x[1]', '0']), V)
    v = df.interpolate(df.Expression(['-x[1]', 'x[0]', 'x[2]']), V)
    w = df.interpolate(df.Expression(['x[1]*x[2]', '-x[0]*x[2]', 'x[0]*x[0]+x[1]*x[1]']), V)

    a = u.vector().array()
    b = v.vector().array()
    c = w.vector().array()

    axb = crossprod(a, b)
    assert(np.allclose(axb, c))


def test_apply_vertexwise():
    xmin = ymin = zmin = -2
    xmax = ymax = zmax = 3
    nx = ny = nz = 10
    mesh = df.BoxMesh(xmin, ymin, zmin, xmax, ymax, zmax, nx, ny, nz)
    V = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
    u = df.interpolate(df.Expression(['x[0]', 'x[1]', '0']), V)
    v = df.interpolate(df.Expression(['-x[1]', 'x[0]', 'x[2]']), V)
    w = df.interpolate(df.Expression(['x[1]*x[2]', '-x[0]*x[2]', 'x[0]*x[0]+x[1]*x[1]']), V)
    #W = df.VectorFunctionSpace(mesh, 'CG', 1)
    #w2 = df.interpolate(df.Expression(['-x[0]*x[1]', 'x[1]*x[0]', '0']), W)

    uxv = apply_vertexwise(np.cross, u, v)
    #udotv = apply_vertexwise(np.dot, u, v)

    assert(np.allclose(uxv.vector().array(), w.vector().array()))
    #assert(np.allclose(udotv.vector().array(), w2.vector().array()))


def test_TemporaryDirectory():
    # Check that the directory is created as expected and destroyed
    # when leaving the with-block.
    with TemporaryDirectory() as tmpdir:
        assert(os.path.exists(tmpdir))
    assert(not os.path.exists(tmpdir))

    # With 'keep=True' the directory should not be deleted.
    with TemporaryDirectory(keep=True) as tmpdir2:
        assert(os.path.exists(tmpdir2))
    assert(os.path.exists(tmpdir2))

    # Tidy up
    os.rmdir(tmpdir2)


def test_contextmanager_ignored(tmpdir):
    d = {}  # dummy dictionary
    s = 'foobar'

    with pytest.raises(KeyError):
        with ignored(OSError):
            d.pop('non_existing_key')

    # This should work because we are ignoring the right kind of error.
    with ignored(KeyError):
        d.pop('non_existing_key')

    # Check that we can ignore multiple errors
    with ignored(IndexError, KeyError):
        d.pop('non_existing_key')
        s[42]


def test_run_cmd_with_timeout():
    # A successfully run command should have exit code 0
    returncode, stdout, _ = run_cmd_with_timeout('echo hello', timeout_sec=100)
    assert(returncode == 0)
    assert stdout == 'hello\n'

    # A non-existing command should raise OSError
    with pytest.raises(OSError):
        returncode, _, _ = run_cmd_with_timeout('foo', timeout_sec=1)

    # This command should be killed due to the timeout, resulting in a return code of -9.
    returncode, _, _ = run_cmd_with_timeout('sleep 10', timeout_sec=0)
    assert(returncode == -9)


@pytest.mark.skipif("True")
def test_jpg2avi(tmpdir):
    """
    Test whether we can create an animation from a series of .jpg images.

    """
    os.chdir(str(tmpdir))
    sim = finmag.example.normal_modes.disk()
    sim.compute_normal_modes(n_values=3)
    sim.export_normal_mode_animation(k=0, filename='foo/bar.pvd')
    render_paraview_scene('foo/bar.pvd', outfile='foo/quux.jpg', trim_border=False)  # note that we must not trim the border because otherwise the resulting .jpg files will have different sizes, which confused mencoder

    # Test the bare-bones export
    jpg2avi('foo/quux.jpg')
    assert(os.path.exists('foo/quux.avi'))

    # Test a few keywords
    jpg2avi('foo/quux.jpg', outfilename='animation.avi', duration=10, fps=10)
    assert(os.path.exists('animation.avi'))

@pytest.mark.skipif("True")
def test_pvd2avi(tmpdir):
    """
    Test whether we can create an animation from the timesteps in a .pvd file.

    """
    os.chdir(str(tmpdir))
    sim = finmag.example.normal_modes.disk()
    sim.compute_normal_modes(n_values=3)
    sim.export_normal_mode_animation(k=0, filename='foo/bar.pvd')

    # Test the bare-bones export
    pvd2avi('foo/bar.pvd')
    assert(os.path.exists('foo/bar.avi'))

    # Test a few keywords
    pvd2avi('foo/bar.pvd', outfilename='animation.avi', duration=10, fps=10, add_glyphs=False, colormap='heated_body')
    assert(os.path.exists('animation.avi'))


if __name__ == '__main__':
    pass
    #test_scalar_valued_dg_function()
    #test_pvd2avi('.')
