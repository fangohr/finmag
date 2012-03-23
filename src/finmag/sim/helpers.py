import numpy as np

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

def normalise(vs, length=1):
    """
    Scale the vectors to the specified length.
    Expects the vectors in a list of the form [[x0, y0, z0], ..., [xn, yn, zn]].

    """
    return np.array([length*v/norm(v) for v in vs])

def fnormalise(ar, length=1):
    """
    Like normalise, except it expects the arguments as an numpy.ndarray like
    dolfin provides, so [x0, ..., xn, y0, ..., yn, z0, ..., zn].

    """
    arr = components(ar)
    arr /= np.sqrt(arr[0]*arr[0] + arr[1]*arr[1] + arr[2]*arr[2])
    return np.append(arr[0],[arr[1],arr[2]]) 

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
    return normalise(vectors, length)

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

def quiver(f, mesh, filename, **kwargs):
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
    q.scene.z_plus_view()
    mlab.axes(figure=figure)
    mlab.savefig(filename)
