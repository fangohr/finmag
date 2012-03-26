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

def quiver(f, mesh, filename, title="", **kwargs):
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
    q.scene.z_plus_view()
    mlab.axes(figure=figure)
    mlab.savefig(filename)

def boxplot(arr, filename, **kwargs):
    import matplotlib.pyplot as plt
    plt.boxplot(list(arr), **kwargs)
    plt.savefig(filename)

def finmag_to_oommf(f, oommf_mesh, dims=1):
    """
    Given a dolfin.Function f and a mesh oommf_mesh as defined in
    finmag.util.oommf.mesh, it will probe the values of f at the coordinates
    of oommf_mesh and return the resulting, oommf_compatible mesh_field.

    """
    f_for_oommf = oommf_mesh.new_field(3)
    for i, (x, y, z) in enumerate(oommf_mesh.iter_coords()):
        if dims == 1:
            f_x, f_y, f_z = f(x)
        else:
            f_x, f_y, f_z = f(x, y, z)
        f_for_oommf.flat[0,i] = f_x
        f_for_oommf.flat[1,i] = f_y
        f_for_oommf.flat[2,i] = f_z
    return f_for_oommf.flat

def stats(arr):
    median  = np.median(arr)
    average = np.mean(arr, axis=1)
    minimum = np.nanmin(arr)
    maximum = np.nanmax(arr)
    spread  = np.std(arr, axis=1)
    stats= "  min, median, max = ({0}, {1} {2}),\n  means = {3}),\n  stds = {4}".format(
            minimum, median, maximum, average, spread)
    return stats

