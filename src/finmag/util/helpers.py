import numpy as np
import dolfin as df
import math
import types
import os

def components(vs):
    """
    For a list of vectors of the form [x0, ..., xn, y0, ..., yn, z0, ..., zn]
    this will return a list of vectors with the shape
    [[x0, ..., xn], [y0, ..., yn], [z0, ..., z1]].

    """
    return vs.view().reshape((3, -1))

def vectors(vs):
    """
    For a list of vectors of the form [x0, ..., xn, y0, ..., yn, z0, ..., zn]
    this will return a list of vectors with the shape
    [[x0, y0, z0], ..., [xn, yn, zn]].

    """
    number_of_nodes = len(vs)/3
    return vs.view().reshape((number_of_nodes, -1), order="F")

def for_dolfin(vs):
    """
    The opposite of the function vectors.

    Takes a list with the shape [[x0, y0, z0], ..., [xn, yn, zn]]
    and returns [x0, ..., xn, y0, ..., yn, z0, ..., zn].
    """
    return rows_to_columns(vs).flatten() 

def norm(v):
    """
    Returns the euclidian norm of a vector in three dimensions.

    """
    return np.sqrt(np.dot(v, v))

def normalise_vectors(vs, length=1):
    """
    Scales each vector in the list `vs` to the specified length.
    `vs` should be a list of vectors of the form [[x0, y0, z0], ..., [xn, yn, zn]].

    """
    return np.array([length*v/norm(v) for v in vs])

def fnormalise(arr, length=1):
    """
    Like normalise_vectors, but expects the arguments as a numpy.ndarray in
    the form that dolfin provides: [x0, ..., xn, y0, ..., yn, z0, ..., zn].

    """
    # If arr happens to be of type int, the calculation below is
    # carried out in integers, and behaves unexpectedly.
    assert arr.dtype not in [np.dtype('int32'),np.dtype('int64')]

    arr = arr.reshape((3, -1))
    arr /= np.sqrt(arr[0]*arr[0] + arr[1]*arr[1] + arr[2]*arr[2] )
    arr = arr.ravel()
    return arr

def angle(v1, v2):
    """
    Returns the angle between two three-dimensional vectors.

    """
    return np.arccos(np.dot(v1, v2) / (norm(v1)*norm(v2)))

def rows_to_columns(arr):
    """
    For an array of the shape [[x1, y1, z1], ..., [xn, yn, zn]]
    returns an array of the shape [[x1, ..., xn],[y1, ..., yn],[z1, ..., zn]].

    """
    return arr.reshape(arr.size, order="F").reshape((3, -1))

def perturbed_vectors(n, direction=[1,0,0], length=1):
    """
    Returns n vectors pointing approximatively in the given direction,
    but with a random displacement. The vectors are normalised to the given
    length. The returned array looks like [[x0, y0, z0], ..., [xn, yn, zn]].

    """
    displacements = np.random.rand(n, 3) - 0.5
    vectors = direction + displacements
    return normalise_vectors(vectors, length)

def read_float_data(filename):
    """
    Reads floats stored in the file filename.

    Needs at least one number which can be cast to float on each line.
    Floats are to be separated by whitespace.
    Returns a list where each entry corresponds to the numbers of a line, which
    means that each entry can be itself a list.

    """
    rows = []
    with open(filename, "r") as f:
        for line in f:
            columns = [float(column) for column in line.strip().split()]
            rows.append(columns)
    return rows

def quiver(f, mesh, filename=None, title="", **kwargs):
    """
    Takes a numpy array of the values of a vector-valued function, defined
    over a mesh (either a dolfin mesh, or one from finmag.util.oommf.mesh)
    and shows a quiver plot of the data.

    Accepts mlab quiver3d keyword arguments as keywords,
    which it will pass down.

    """
    assert isinstance(f, np.ndarray)

    from mayavi import mlab
    from dolfin.cpp import Mesh as dolfin_mesh
    from finmag.util.oommf.mesh import Mesh as oommf_mesh
    
    if isinstance(mesh, dolfin_mesh):
        coords = mesh.coordinates()
    elif isinstance(mesh, oommf_mesh):
        coords = np.array(list(mesh.iter_coords()))
    elif isinstance(mesh, np.ndarray) or isinstance(mesh, list):
        # If you know what the data has to look like, why not
        # be able to pass it in directly.
        coords = mesh
    else:
        raise TypeError("Don't know what to do with mesh of class {0}.".format(
            mesh.__class__))

    r = coords.reshape(coords.size, order="F").reshape((coords.shape[1], -1))
    # All 3 coordinates of the mesh points must be known to the plotter,
    # even if the mesh is one-dimensional. If not all coordinates are known,
    # fill the rest up with zeros.
    codimension = 3 - r.shape[0]
    if codimension > 0:
        r = np.append(r, [np.zeros(r[0].shape[0])]*codimension, axis=0)

    if f.size == f.shape[0]:
        # dolfin provides a flat numpy array, but we would like
        # one with the x, y and z components as individual arrays.
        f = components(f)
   
    figure = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    q = mlab.quiver3d(*(tuple(r)+tuple(f)), figure=figure, **kwargs)
    q.scene.isometric_view()
    mlab.title(title)
    mlab.axes(figure=figure)

    if filename:
        mlab.savefig(filename)
    else:
        mlab.show()
    mlab.close(all=True)

def boxplot(arr, filename, **kwargs):
    import matplotlib.pyplot as plt
    plt.boxplot(list(arr), **kwargs)
    plt.savefig(filename)

def stats(arr, axis=1):
    median  = np.median(arr)
    average = np.mean(arr)
    minimum = np.nanmin(arr)
    maximum = np.nanmax(arr)
    spread  = np.std(arr)
    stats= "    min, median, max = {0}, {1}, {2}\n    mean, std = {3}, {4}".format(
            minimum, median, maximum, average, spread)
    return stats

def frexp10(x, method="string"):
    """
    Same as math.frexp but in base 10, will return (m, e)
    such that x == m * 10 ** e.

    """
    if method=="math":
        lb10 = math.log10(x)
        return 10 ** (lb10 - int(lb10)), int(lb10)
    else:
        nb = ("%e" % x).split("e")
        return float(nb[0]), int(nb[1])

def tex_sci(x, p=2):
    """
    Returns LaTeX code for the scientific notation of a floating point number.

    """
    m, e = frexp10(x)
    return "{:.{precision}f} \\times 10^{{{}}}".format(m, e, precision=p)

def sphinx_sci(x, p=2):
    """
    Returns the code you need to have math mode in sphinx and the
    scientific notation of the floating point nunber x.

    """
    return ":math:`{}`".format(tex_sci(x, p))

def normed_func(value, Space, **kwargs):
    if not isinstance(value, tuple):
        raise NotImplementedError, "We only support tuples."

    if isinstance(value[0], str):
        mgen = df.Expression(value, **kwargs)
    else:
        mgen = df.Constant(value)

    m0 = df.interpolate(mgen, Space)
    m0_vec = m0.vector().array().reshape((3, -1))
    m0_vec /= np.sqrt(m0_vec[0]**2 + m0_vec[1]**2 + m0_vec[2]**2)
    m0.vector()[:] = m0_vec.reshape(-1)

    return m0

def normalize_filepath(dirname, filename):
    """
    Return a pair (dirname_new, filename_new) which describes the same
    path as (dirname, filename), but where `dirname_new` contains all
    directory components (i.e., any directory components present in
    `filename` are copied over). Both `dirname_new` and `filename_new`
    are guaranteed to be strings (with `dirname_new` possibly being
    empty).

    Examples:

        normalize_filepath(None, 'foo/bar.txt') = ('foo', 'bar.txt')
        normalize_filepath('foo', 'bar/baz.txt') = ('foo/bar', 'baz.txt')

    `dirname` may be None or any string (including the empty string).
    `filename` must be a non-empty string.

    `filename` is allowed to contain directory components, but if
    `directory` is specified (i.e., neither None nor the empty string)
    then `filename` must not describe an absolute path.
    """
    if not isinstance(dirname, (str, types.NoneType)):
        raise TypeError("'dirname' must be a string or None (got: '%s' of type '%s')".format(dirname, type(dirname)))
    if not isinstance(filename, str):
        raise TypeError("'filename' must be a string (got: '%s' of type '%s')".format(filename, type(filename)))
    if filename == '':
        raise ValueError("'filename' must not be empty")

    if os.path.isabs(filename) and dirname not in (None, ''):
        raise ValueError("If 'dirname' is specified the 'filename' must not contain an absolute path. Arguments given: dirname='{}', filename='{}'".format(dirname, filename))

    full_name = os.path.join(dirname or "", filename)
    d = os.path.dirname(full_name)
    f = os.path.basename(full_name)
    return (d, f)
