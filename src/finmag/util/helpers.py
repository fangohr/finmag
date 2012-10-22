import matplotlib.pyplot as plt
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
    Returns a new list containing the vectors in `vs`, scaled to the specified length.

    `vs` should be a list of vectors of the form [[x0, y0, z0], ..., [xn, yn, zn]].

    """
    return np.array([length*v/norm(v) for v in vs])

def fnormalise(arr):
    """
    Like normalise_vectors, but expects the arguments as a numpy.ndarray in
    the form that dolfin provides: [x0, ..., xn, y0, ..., yn, z0, ..., zn].

    """
    # If arr happens to be of type int, the calculation below is
    # carried out in integers, and behaves unexpectedly.
    assert arr.dtype not in [np.dtype('int32'),np.dtype('int64')]

    arr = arr.reshape((3, -1)).copy()
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

def vector_valued_function(value, S3, normalise=False, **kwargs):
    """
    Create a constant function on the VectorFunctionSpace `S3` whose value
    is the 3-vector `value`. Returns an object of type 'df.Function'.

    `value` can be any of the following:

        - tuple, list or numpy.ndarray of length 3

        - dolfin.Constant representing a 3-vector

        - 3-tuple of strings (with keyword arguments if needed),
          which will get cast to a dolfin.Expression where any variables in
          the expression are substituted with the values taken from 'kwargs'

        - numpy.ndarray of nodal values of the shape (3*n,), where n
          is the number of nodes

        - function (any callable object will do) which accepts the
          coordinates of all mesh nodes as a numpy.ndarray of shape (3, n)
          and returns the field H in this form as well


    *Arguments*

       value     -- the value of the function

       S3        -- dolfin.VectorFunctionSpace of dimension 3

       normalize -- if True then the function values are normalised to
                    unit length (default: False)

       kwargs    -- if `value` is a 3-tuple of strings (which will be
                    cast to a dolfin.Expression), then any variables
                    occurring in them will be substituted with the
                    values in kwargs; otherwise kwargs is ignored

    """
    def _const_function(value, S3):
        # Filling the dolfin.Function.vector() directly is about two
        # orders of magnitudes faster than using df.interpolate()!
        #
        val = np.empty((S3.mesh().num_vertices(), 3))
        val[:] = value  # fill the array with copies of 'value' (we're using broadcasting here!)
        fun = df.Function(S3)
        fun.vector()[:] = val.transpose().reshape(-1) # transpose is necessary because of the way dolfin aligns the function values internally
        return fun

    if isinstance(value, (tuple, list)):
        if isinstance(value[0], str):
            # a tuple of strings is considered to be the ingredient
            # for a dolfin expression, whereas a tuple of numbers
            # would signify a constant
            val = df.Expression(value, **kwargs)
            fun = df.interpolate(val, S3)
        else:
            fun = _const_function(value, S3)
    elif isinstance(value, (df.Constant, df.Expression)):
        fun = df.interpolate(value, S3)
    elif isinstance(value, np.ndarray):
        if len(value) == 3:
            fun = _const_function(value, S3)
        else:
            fun = df.Function(S3)
            fun.vector()[:] = value
    elif hasattr(value, '__call__'):
        coords = np.array(zip(* S3.mesh().coordinates()))
        fun = df.Function(S3)
        fun.vector()[:] = value(coords).flatten()
    else:
        raise AttributeError

    if normalise:
        fun.vector()[:] = fnormalise(fun.vector().array())

    return fun

def plot_hysteresis_loop(H_vals, m_vals, style='o-', add_point_labels=False, infobox=[], infobox_loc='bottom right',
                         filename=None, title="Hysteresis loop", xlabel="H_ext (A/m)", ylabel="m", figsize=(10, 7)):
    """
    Produce a hysteresis plot

    Arguments:

       H_vals -- list of scalar values; the values of the applied field used for the relaxation
                 stages of the hysteresis loop

       m_vals -- list of scalar values; the magnetisation obtained at the end of each relaxation
                 stage in the hysteresis loop

    Keyword arguments:

       style -- the plot style (default: 'o-')

       add_point_labels -- if True (default: False), every point is labeled with a number which
                           indicates the relaxation stage of the hysteresis loop it represents

       infobox -- list; each entry can be either a string or a pair of the form (name, value).
                  If not empty, an info box will added to the plot with the list elements appearing
                  on separate lines. Strings are printed verbatim, whereas name/value pairs are
                  converted to a string of the form "name = value".

       filename -- if given, save the resulting plot to a file with the specified name
    """
    if not all([isinstance(x, (types.IntType, types.FloatType)) for x in m_vals]):
        raise ValueError("m_vals must be a list of scalar values, got: {}".format(m_vals))

    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    N = len(H_vals) // 2
    N_vals = range(N,0,-1) + range(1,N+1)
    H_max = max(H_vals)

    ax.plot(H_vals, m_vals, style)

    ax.set_xlim(-1.1*H_max, 1.1*H_max)
    ax.set_ylim((-1.2, 1.2))

    if add_point_labels:
        for i in xrange(len(H_vals)):
            x = H_vals[i]
            y = m_vals[i]
            ax.annotate(str(i), xy=(x, y), xytext=(-10, 5) if i<N else (0, -15), textcoords='offset points')

    # draw the info box
    if infobox != []:
        box_text = ""
        for elt in infobox:
            if isinstance(elt, types.StringType):
                box_text += elt+'\n'
            else:
                try:
                    name, value = elt
                    box_text += "{} = {}\n".format(name, value)
                except ValueError:
                    raise ValueError, "All list elements in 'infobox' must be either strings or pairs of the form (name, value). Got: '{}'".format(elt)
        box_text = box_text.rstrip()

        if infobox_loc not in ["top left", "top right", "bottom left", "bottom right"]:
            raise ValueError("'infobox_loc' must be one of 'top left', 'top right', 'bottom left', 'bottom right'.")

        vpos, hpos = infobox_loc.split()
        x = H_max if hpos == "right" else -H_max
        y = 1.0 if vpos == "top" else -1.0

        ax.text(x, y, box_text, size=12,
                horizontalalignment=hpos, verticalalignment=vpos, multialignment="left", #transform = ax.transAxes,
                bbox=dict(boxstyle="round, pad=0.3", facecolor="white", edgecolor="green", linewidth=1))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()

    if filename:
        fig.savefig(filename)
