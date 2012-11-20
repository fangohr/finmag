from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import subprocess
import logging
import numpy as np
import dolfin as df
import math
import types
import sys
import os
from finmag.util.meshes import mesh_volume
from math import sqrt, pow

logger = logging.getLogger("finmag")

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

def norm(vs):
    """
    Returns the euclidian norm of one or several vectors in three dimensions.

    When passing an array of vectors, the shape is expected to be in the form
    [[x0, y0, z0], ..., [xn, yn, zn]].

    """
    if not type(vs) == np.ndarray:
        vs = np.array(vs)
    if vs.shape == (3, ):
        return np.linalg.norm(vs)
    return np.sqrt(np.add.reduce(vs*vs, axis=1))

def fnormalise(arr):
    """
    Returns a normalised copy of the vectors in arr.
    
    Expects arr to be a numpy.ndarray in the form that dolfin
    provides: [x0, ..., xn, y0, ..., yn, z0, ..., zn].

    """
    a = arr.astype(np.float64) # this copies

    a = a.reshape((3, -1))
    a /= np.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
    a = a.ravel()
    return a

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

def _create_nonexistent_directory_components(filename):
    """
    Creates any directory components in 'filename' which don't exist yet.
    For example, if filename='/foo/bar/baz.txt' then the directory /foo/bar
    will be created.
    """
    # Create directory part if it does not exist
    dirname = os.path.dirname(filename)
    if dirname != '':
        if not os.path.exists(dirname):
            os.makedirs(dirname)


def plot_hysteresis_loop(H_vals, m_vals, style='x-', add_point_labels=False, point_labels=None, infobox=[], infobox_loc='bottom right',
                         filename=None, title="Hysteresis loop", xlabel="H_ext (A/m)", ylabel="m_avg", figsize=(10, 7)):
    """
    Produce a hysteresis plot

    Arguments:

       H_vals -- list of scalar values; the values of the applied field used for the relaxation
                 stages of the hysteresis loop

       m_vals -- list of scalar values; the magnetisation obtained at the end of each relaxation
                 stage in the hysteresis loop

    Keyword arguments:

       style -- the plot style (default: 'x-')

       add_point_labels -- if True (default: False), every point is labeled with a number which
                           indicates the relaxation stage of the hysteresis loop it represents

       point_labels -- list or None; each list element can either be an integer or a pair of the
                       form (index, label), where index is an integer and label is a string. If
                       not None, only the points whose index appears in this list are labelled
                       (either with their index, or with the given label string if provided).
                       For example, if only every 10th point should be labeled, one might say
                       'point_labels=xrange(0, NUM_POINTS, 10)'.

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
    H_max = max(H_vals)

    ax.plot(H_vals, m_vals, style)

    ax.set_xlim(-1.1*H_max, 1.1*H_max)
    ax.set_ylim((-1.2, 1.2))

    if point_labels is None:
        point_labels = xrange(len(H_vals))
    # Convert point_labels into a dictionary where the keys are the point indices
    # and the values are the respective labels to be used.
    point_labels = dict(map(lambda i: i if isinstance(i, tuple) else (i, str(i)), point_labels))
    if add_point_labels:
        for i in xrange(len(H_vals)):
            if point_labels.has_key(i):
                x = H_vals[i]
                y = m_vals[i]
                ax.annotate(point_labels[i], xy=(x, y), xytext=(-10, 5) if i<N else (0, -15), textcoords='offset points')

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
        _create_nonexistent_directory_components(filename)
        fig.savefig(filename)

def duplicate_output_to_file(filename, add_timestamp=False, timestamp_fmt='__%Y-%m-%d_%H.%M.%S'):
    """
    Redirect all (future) output to a file with the given filename.
    This redirects both to stdout and stderr.

    If `add_timestamp` is True then a timestamp will be added to the
    filename indicating the time when the call to this function
    occurred (for example, the filename 'output.txt' might be changed
    into 'output_2012-01-01_14.33.52.txt'). The timestamp format can
    be controlled via `timestamp_fmt`, which should be a formatting
    string as accepted by `datetime.strftime`.
    """
    _create_nonexistent_directory_components(filename)
    if add_timestamp:
        name, ext = os.path.splitext(filename)
        filename = '{}{}{}'.format(name, datetime.strftime(datetime.now(), timestamp_fmt), ext)

    logger.debug("Duplicating output to file '{}'".format(filename))
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    tee = subprocess.Popen(["tee", filename], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def cartesian_to_spherical(vector):
    """
    Converts cartesian coordinates to spherical coordinates.

    Returns a tuple (r, theta, phi) where r is the radial distance, theta
    is the inclination (or elevation) and phi is the azimuth (ISO standard 31-11).

    """
    r = np.linalg.norm(vector)
    unit_vector = np.array(vector) / r
    theta = np.arccos(unit_vector[2])
    phi = np.arctan2(unit_vector[1], unit_vector[0])
    return np.array((r, theta, phi))

def spherical_to_cartesian(v):
    """
    Converts spherical coordinates to cartesian.

    Expects the arguments r for radial distance, inclination theta
    and azimuth phi.

    """
    r, theta, phi = v
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array((x, y, z))

def pointing_upwards((x, y, z)):
    """
    Returns a boolean that is true when the vector is pointing upwards.
    Upwards is defined as having a polar angle smaller than 45 degrees.

    """
    _, theta, _ = cartesian_to_spherical((x, y, z))
    return theta <= (np.pi / 4)

def pointing_downwards((x, y, z)):
    """
    Returns a boolean that is true when the vector is pointing downwards.
    Downwards is defined as having a polar angle between 135 and 225 degrees.

    """
    _, theta, _ = cartesian_to_spherical((x, y, z))
    return abs(theta - np.pi) < (np.pi / 4) 


def mesh_functions_allclose(f1, f2, fun_mask=None, rtol=1e-05, atol=1e-08):
    """
    Return True if the values and `f1` and `f2` are nearly identical at all
    mesh nodes outside a given region.

    The region to be ignored is defined by `fun_mask`. This should be a
    function which accept a mesh node (= a tuple or list of coordinates) as its
    single argument and returns True or False, depending on whether the node
    lies within the region to be masked out. The values of `f1` and `f2` are
    then ignored at nodes where `fun_mask` evaluates to True (if `fun_mask` is
    None, all node values are taken into account).

    Note that the current implementation assumes that the degrees of freedom
    of both functions are at the mesh nodes (in particular, f1 and f2 should
    not be elements of a Discontinuous Galerkin function space, for example).

    *Arguments*

    f1, f2 : dolfin Functions
        The functions to be compared.

    fun_mask : None or dolfin Function
        Indicator function of the region which should be ignored.

    rtol, atol : float
        The relative/absolute tolerances used for comparison. These have the same
        meaning as for numpy.allclose().
    """
    # XXX FIXME: This is a very crude implementation for now. It should be refined
    #            once I understand how to properly deal with dolfin Functions, in
    #            particular how to use the product of funtions.
    if f1.function_space() != f2.function_space():
        raise ValueError("Both functions must be defined on the same FunctionSpace")
    V = f1.function_space()
    pts = V.mesh().coordinates()

    if fun_mask is None:
        mask = np.ones(len(pts))
    else:
        mask = np.array(map(lambda pt: 0.0 if fun_mask(pt) else 1.0, pts))

    v1 = f1.vector()
    v2 = f2.vector()
    return np.allclose(np.fabs(v1*mask - v2*mask), 0.0, rtol=rtol, atol=atol)


def piecewise_on_subdomains(mesh, mesh_function, fun_vals):
    """
    Constructs and returns a dolfin Function which is piecewise constant on
    certain subdomains of a mesh.

    *Arguments*

    mesh : dolfin Mesh
        The mesh on which the new function will be defined.

    mesh_function : dolfin MeshFunction
        A function assigning to each cell the subdomain number to
        which this cell belongs.

    fun_vals : sequence
        The function values for the returned function, where the first
        value provided corresponds to the first region defined in the mesh
        and so on.

    *Returns*

    A dolfin function which is piecewise constant on subdomains and on the
    subdomain with index `idx` assumes the value given by `fun_vals[idx]`.
    """
    V = df.FunctionSpace(mesh, 'DG', 0)
    f = df.Function(V)

    help = np.asarray(mesh_function.array() - 1, dtype=np.int32)
    f.vector()[:] = np.choose(help, fun_vals)
    return f
