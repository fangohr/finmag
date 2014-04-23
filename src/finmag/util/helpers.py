from __future__ import division
from datetime import datetime
from glob import glob
from contextlib import contextmanager
from finmag.util.fileio import Tablereader
from finmag.util.visualization import render_paraview_scene
from finmag.util.versions import get_version_dolfin
from finmag.util import ansistrm
from threading import Timer
from distutils.version import LooseVersion
import matplotlib as mpl
#try to use 'Agg' as backend, we can remove it if something wrong.
#mpl.use("Agg")
#this is the first place to import pyplot in finmag
import matplotlib.pyplot as plt
import subprocess as sp
import shlex
import itertools
import tempfile
import shutil
import logging.handlers
import numpy as np
import dolfin as df
import math
import types
import sys
import os
import re
import sh

logger = logging.getLogger("finmag")


def create_missing_directory_components(filename):
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


def logging_handler_str(handler):
    """
    Return a string describing the given logging handler.

    """
    if handler.__class__ == logging.StreamHandler:
        handlerstr = str(handler.stream)
    elif handler.__class__ in [logging.FileHandler, logging.handlers.RotatingFileHandler]:
        handlerstr = str(handler.baseFilename)
    else:
        handlerstr = str(handler)
    return handlerstr


def logging_status_str():
    """
    Return a string that shows all known loggers and their current levels.
    This is useful for debugging of the logging module.

    """
    rootlog = logging.getLogger('')
    msg = ("Current logging status: "
           "rootLogger level=%2d\n" % rootlog.level)

    # This keeps the loggers (with the exception of root)
    loggers = logging.Logger.manager.loggerDict
    for loggername, logger in [('root', rootlog)] + loggers.items():
        for i, handler in enumerate(logger.handlers):
            handlerstr = logging_handler_str(handler)
            msg += (" %15s (lev=%2d, eff.lev=%2d) -> handler %d: lev=%2d %s\n"
                    % (loggername, logger.level, logger.getEffectiveLevel(),
                       i, handler.level, handlerstr))
    return msg


def set_logging_level(level):
    """
    Set the level for finmag log messages.

    *Arguments*

    level: string

       One of the levels supported by Python's `logging` module.
       Supported values: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL' and
       the finmag specific level 'EXTREMEDEBUG'.
    """
    if level not in ['EXTREMEDEBUG', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        raise ValueError("Logging level must be one of: 'DEBUG', 'INFO', "
                         "'WARNING', 'ERROR', 'CRITICAL'")
    logger.setLevel(level)


supported_color_schemes = ansistrm.level_maps.keys()
supported_color_schemes_str = ", ".join(["'{}'".format(s) for s in supported_color_schemes])

def set_color_scheme(color_scheme):
    """
    Set the color scheme for finmag log messages in the terminal.

    *Arguments*

    color_scheme: string

       One of the color schemes supported by Python's `logging` module.
       Supported values: {}.

    """
    if color_scheme not in supported_color_schemes:
        raise ValueError("Color scheme must be one of: {}".format(supported_color_schemes_str))
    for h in logger.handlers:
        if not isinstance(h, ansistrm.ColorizingStreamHandler):
            continue
        h.level_map = ansistrm.level_maps[color_scheme]
# Insert supported color schemes into docstring
set_color_scheme.__doc__ = set_color_scheme.__doc__.format(supported_color_schemes_str)


def start_logging_to_file(filename, formatter=None, mode='a', level=logging.DEBUG, rotating=False, maxBytes=0, backupCount=1):
    """
    Add a logging handler to the "finmag" logger which writes all
    (future) logging output to the given file. It is possible to call
    this multiple times with different filenames. By default, if the
    file already exists then new output will be appended at the end
    (use the 'mode' argument to change this).

    *Arguments*

    formatter: instance of logging.Formatter

        For details, see the section 'Formatter Objectsion' in the
        documentation of the logging module.

    mode: ['a' | 'w']

        Determines whether new content is appended at the end ('a') or
        whether logfile contents are overwritten ('w'). Default: 'a'.

    rotating: bool

        If True (default: False), limit the size of the logfile to
        `maxBytes` (0 means unlimited). Once the file size is near
        this limit, a 'rollover' will occur. See the docstring of
        `logging.handlers.RotatingFileHandler` for details.

    *Returns*

    The newly created logging hander is returned.
    """
    if formatter is None:
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')

    filename = os.path.abspath(os.path.expanduser(filename))
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    h = logging.handlers.RotatingFileHandler(filename, mode=mode, maxBytes=maxBytes, backupCount=backupCount)
    h.setLevel(level)
    h.setFormatter(formatter)
    if mode == 'a':
        logger.info("Finmag logging output will be appended to file: "
                    "'{}'".format(filename))
    else:
        # XXX FIXME: There is still a small bug here: if we create
        # multiple simulations with the same name from the same
        # ipython session, the logging output of the first will not be
        # deleted. For example:
        #
        #    from finmag import sim_with
        #    import dolfin as df
        #    import logging
        #
        #    logger = logging.getLogger("finmag")
        #    mesh = df.BoxMesh(0, 0, 0, 1, 1, 1, 5, 5, 5)
        #
        #    logger.debug("Creating first simulation")
        #    sim = sim_with(mesh, 1e6, m_init=(1,0,0), name="sim1")
        #
        #    logger.debug("Creating second simulation")
        #    sim = sim_with(mesh, 1e6, m_init=(1,0,0), name="sim1")
        #
        # At the end the output of the first simulation is still
        # present in the logfile "sim1.log".
        logger.info("Finmag logging output will be written to file: '{}' "
                    "(any old content will be overwritten).".format(filename))
    logger.addHandler(h)
    return h


def get_hg_revision_info(repo_dir, revision='tip'):
    """
    Returns the revision number, revision ID and date of a revision in
    the given Mercurial repository. For example, the information
    returned might be:

        (3486, '18e7def5e18a', '2013-04-25')

    """
    cwd_bak = os.getcwd()
    try:
        os.chdir(os.path.expanduser(repo_dir))
    except OSError:
        raise ValueError("Expected a valid repository, but directory does not exist: '{}'".format(repo_dir))

    try:
        rev_nr = int(sp.check_output(['hg', 'id', '-n', '-r', revision]).strip())
        rev_id = sp.check_output(['hg', 'id', '-i', '-r', revision]).strip()
        #rev_log = sp.check_output(['hg', 'log', '-r', revision]).strip()
        rev_date = sp.check_output(['hg', 'log', '-r', revision, '--template', '{date|isodate}']).split()[0]
    except sp.CalledProcessError:
        raise ValueError("Invalid revision '{}', or invalid Mercurial repository: '{}'".format(revision, repo_dir))

    os.chdir(cwd_bak)
    return rev_nr, rev_id, rev_date


def binary_tarball_name(repo_dir, revision='tip', suffix=''):
    """
    Returns the name of the Finmag binary tarball if built from the
    given repository and revision.

    The general pattern is something like:

       FinMag-dist__2013-04-25__rev3486_18e7def5e18a_suffix.tar.bz2

    If specified, the suffix is inserted immediately before '.tar.bz2'.


    *Arguments*

    repo_dir :  name of a directory containing a valid Finmag repository.

    revision :  the revision to be bundled in the tarball.

    suffix :  string to be appended to the tarball

    """
    # XXX TODO: Should we also check whether the repo is actually a Finmag repository?!?
    rev_nr, rev_id, rev_date = get_hg_revision_info(repo_dir, revision)
    tarball_name = "FinMag-dist__{}__rev{}_{}{}.tar.bz2".format(rev_date, rev_nr, rev_id, suffix)
    return tarball_name


def clean_filename(filename):
    """
    Remove non-alphanumeric characters from filenames.

    *Parameters*

    filename : str
        The filename to be sanitized.

    *Returns*

    clean : str
        A sanitized filename that contains only alphanumeric
        characters and underscores.
    """
    filename = re.sub(r'[^a-zA-Z0-9_]', '_', filename)
    return filename


def assert_number_of_files(files, n):
    """
    Check that there are exactly `n` files matching the pattern in
    `files` (which may contain wildcards, such as 'foo/bar*.txt') and
    raise an AssertionError otherwise.
    """
    assert(len(glob(files)) == n)

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


def crossprod(v, w):
    """
    Compute the point-wise cross product of two (3D) vector fields on a mesh.

    The arguments `v` and `w` should be numpy.arrays representing
    dolfin functions, i.e. they should be of the form [x0, ..., xn,
    y0, ..., yn, z0, ..., zn]. The return value is again a numpy array
    of the same form.

    """
    if df.parameters.reorder_dofs_serial != False:
        raise RuntimeError("Please ensure that df.parameters.reorder_dofs_serial is set to False.")
    assert(v.ndim == 1 and w.ndim == 1)
    a = v.reshape(3, -1)
    b = w.reshape(3, -1)
    return np.cross(a, b, axisa=0, axisb=0, axisc=0).reshape(-1)


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

def mesh_and_space(mesh_or_space):
    """
    Return a (df.Mesh, df.VectorFuntionspace) tuple where one of the two items
    was passed in as argument and the other one built/extracted from it.

    """
    if isinstance(mesh_or_space, df.VectorFunctionSpace):
        S3 = mesh_or_space
        mesh = S3.mesh()
    else:
        mesh = mesh_or_space
        S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    return mesh, S3

def mesh_equal(mesh1,mesh2):
    cds1=mesh1.coordinates()
    cds2=mesh2.coordinates()
    return np.array_equal(cds1,cds2)

# TODO: In dolfin 1.2 if the pbc are used, the degree of freedom for functionspace
#       is different from the number of  mesh coordinates, so we need to consider this
#       problem as well
def vector_valued_function(value, mesh_or_space, normalise=False, **kwargs):
    """
    Create a vector-valued function on the given mesh or VectorFunctionSpace.
    Returns an object of type 'df.Function'.

    `value` can be any of the following:

        - tuple, list or numpy.ndarray of length 3

        - dolfin.Constant representing a 3-vector

        - dolfin.Expression

        - 3-tuple of strings (with keyword arguments if needed),
          which will get cast to a dolfin.Expression where any variables in
          the expression are substituted with the values taken from 'kwargs'

        - numpy.ndarray of nodal values of the shape (3*n,), where n
          is the number of nodes. Note that the elements in this array
          should follow dolfin's convention (i.e., the x-coordinates
          of all function values should be listed first, then the y-
          and z-values). The shape can also be (n, 3) with one vector
          per node.

        - function (any callable object will do) of the form:

             f: (x, y, z) -> v

          where v is the 3-vector that is the function value at the
          point (x, y, z).


    *Arguments*

       value            -- the value of the function as described above

       mesh_or_space    -- either a dolfin.VectorFunctionSpace of dimension 3
                           or a dolfin.Mesh

       normalise        -- if True then the function values are normalised to
                           unit length (default: False)

       kwargs    -- if `value` is a 3-tuple of strings (which will be
                    cast to a dolfin.Expression), then any variables
                    occurring in them will be substituted with the
                    values in kwargs; otherwise kwargs is ignored

    """
    mesh, S3 = mesh_and_space(mesh_or_space)

    if isinstance(value, (df.Constant, df.Expression)):
        fun = df.interpolate(value, S3)
    elif isinstance(value, (tuple, list, np.ndarray)) and len(value) == 3:
        # We recognise a sequence of strings as ingredient for a df.Expression.
        if all(isinstance(item, basestring) for item in value):
            expr = df.Expression(value, **kwargs)
            fun = df.interpolate(expr, S3)
        else:
            fun = df.Function(S3)
            vec = np.empty((fun.vector().size()/3, 3))
            vec[:] = value # using broadcasting

            fun.vector().set_local(vec.transpose().reshape(-1))
    elif isinstance(value, np.ndarray):
        fun = df.Function(S3)
        if value.ndim == 2:
            assert value.shape[1] == 3
            value = value.reshape(value.size, order="F")
        fun.vector()[:] = value

    #if it's a normal function, we wrapper it into a dolfin expression
    elif hasattr(value, '__call__'):

        class HelperExpression(df.Expression):
            def __init__(self,value):
                super(HelperExpression, self).__init__()
                self.fun = value

            def eval(self, value, x):
                value[:] = self.fun(x)[:]

            def value_shape(self):
                return (3,)

        hexp = HelperExpression(value)
        fun = df.interpolate(hexp, S3)

    else:
        raise TypeError("Cannot set value of vector-valued function from "
                        "argument of type '{}'".format(type(value)))

    if normalise:
        fun.vector()[:] = fnormalise(fun.vector().array())

    return fun

# XXX TODO: This should perhaps be merged with scalar_valued_dg_function
# to avoid code duplication (but only if it doesn't obfuscate the
# interface and it is clear how to distinguish whether a Lagrange or
# DG function space should be used).
def scalar_valued_function(value, mesh_or_space):
    """
    Create a scalar function on the given mesh or VectorFunctionSpace.

    If mesh_or_space is a FunctionSpace, it should be of type "Lagrange"
    (for "DG" spaces use the function `scalar_valued_dg_function`
    instead). Returns an object of type 'df.Function'.

    `value` can be any of the following (see `vector_valued_function`
    for more details):

        - a number

        - numpy.ndarray or a common list

        - dolfin.Constant or dolfin.Expression

        - function (or any callable object)
    """
    if isinstance(mesh_or_space, df.FunctionSpace):
        S1 = mesh_or_space
        mesh = S1.mesh()
    else:
        mesh = mesh_or_space
        S1 = df.FunctionSpace(mesh, "Lagrange", 1)

    if isinstance(value, (df.Constant, df.Expression)):
        fun = df.interpolate(value, S1)
    elif isinstance(value, (np.ndarray,list)):
        fun = df.Function(S1)
        assert(len(value)==fun.vector().size())
        fun.vector().set_local(value)
    elif isinstance(value,(int,float,long)):
        fun = df.Function(S1)
        fun.vector()[:]=value
    elif hasattr(value, '__call__'):

        #if it's a normal function, we wrapper it into a dolfin expression
        class HelperExpression(df.Expression):
            def __init__(self,value):
                super(HelperExpression, self).__init__()
                self.fun = value

            def eval(self, value, x):
                value[0] = self.fun(x)

        hexp = HelperExpression(value)
        fun = df.interpolate(hexp, S1)

    else:
        raise TypeError("Cannot set value of scalar-valued function from "
                        "argument of type '{}'".format(type(value)))

    return fun


def scalar_valued_dg_function(value, mesh_or_space):
    """
    Create a scalar function on the given mesh or VectorFunctionSpace.

    If mesh_or_space is a FunctionSpace, it should be of type "DG"
    (for "Lagrange" spaces use the function `scalar_valued_function`
    instead). Returns an object of type 'df.Function'.

    `value` can be any of the following (see `vector_valued_function`
    for more details):

        - a number

        - numpy.ndarray or a common list

        - dolfin.Constant or dolfin.Expression

        - function (or any callable object)
    """
    if isinstance(mesh_or_space, df.FunctionSpace):
        dg = mesh_or_space
        mesh = dg.mesh()
    else:
        mesh = mesh_or_space
        dg = df.FunctionSpace(mesh, "DG", 0)

    if isinstance(value, (df.Constant, df.Expression)):
        fun = df.interpolate(value, dg)
    elif isinstance(value, (np.ndarray,list)):
        fun = df.Function(dg)
        assert(len(value)==fun.vector().size())
        fun.vector().set_local(value)
    elif isinstance(value,(int,float,long)):
        fun = df.Function(dg)
        fun.vector()[:]=value
    elif isinstance(value, df.Function):
        mesh1=value.function_space().mesh()
        fun = df.Function(dg)
        if mesh_equal(mesh,mesh1) and value.vector().size()==fun.vector().size():
            fun=value
        else:
            raise RuntimeError("Meshes are not compatible for given function.")
    elif hasattr(value, '__call__'):

        class HelperExpression(df.Expression):
            def __init__(self,value):
                super(HelperExpression, self).__init__()
                self.fun = value

            def eval(self, value, x):
                value[0] = self.fun(x)

        hexp = HelperExpression(value)
        fun = df.interpolate(hexp, dg)

    else:
        raise TypeError("Cannot set value of scalar-valued DG function from "
                        "argument of type '{}'".format(type(value)))

    return fun


def vector_valued_dg_function(value, mesh_or_space, normalise=False):
    """
    Create a vector function on the given mesh or VectorFunctionSpace.

    If mesh_or_space is a FunctionSpace, it should be of type "DG"
    (for "Lagrange" spaces use the function `scalar_valued_function`
    instead). Returns an object of type 'df.Function'.

    `value` can be any of the following (see `vector_valued_function`
    for more details):

        - a number

        - numpy.ndarray or a common list

        - dolfin.Constant or dolfin.Expression

        - function (or any callable object)
    """
    if isinstance(mesh_or_space, df.VectorFunctionSpace):
        dg = mesh_or_space
        mesh = dg.mesh()
    else:
        mesh = mesh_or_space
        dg = df.VectorFunctionSpace(mesh, "DG", 0)

    if isinstance(value, (df.Constant, df.Expression)):
        fun = df.interpolate(value, dg)
    elif isinstance(value, (tuple, list, np.ndarray)) and len(value) == 3:
        # We recognise a sequence of strings as ingredient for a df.Expression.
        if all(isinstance(item, basestring) for item in value):
            expr = df.Expression(value, )
            fun = df.interpolate(expr, dg)
        else:
            fun = df.Function(dg)
            vec = np.empty((fun.vector().size()/3, 3))
            vec[:] = value # using broadcasting

            fun.vector().set_local(vec.transpose().reshape(-1))
    elif isinstance(value, (np.ndarray,list)):
        fun = df.Function(dg)
        assert(len(value)==fun.vector().size())
        fun.vector().set_local(value)
    elif isinstance(value,(int,float,long)):
        fun = df.Function(dg)
        fun.vector()[:]=value
    elif isinstance(value, df.Function):
        mesh1=value.function_space().mesh()
        fun = df.Function(dg)
        if mesh_equal(mesh,mesh1) and value.vector().size()==fun.vector().size():
            fun=value
        else:
            raise RuntimeError("Meshes are not compatible for given function.")
    elif hasattr(value, '__call__'):

        class HelperExpression(df.Expression):
            def __init__(self,value):
                super(HelperExpression, self).__init__()
                self.fun = value

            def eval(self, value, x):
                value[:] = self.fun(x)[:]

            def value_shape(self):
                return (3,)

        hexp = HelperExpression(value)
        fun = df.interpolate(hexp, dg)

    else:
        raise TypeError("Cannot set value of vector-valued DG function from "
                        "argument of type '{}'".format(type(value)))

    if normalise:
        fun.vector()[:] = fnormalise(fun.vector().array())

    return fun


def value_for_region(mesh, value, region_no, default_value=0, project_to_CG=False):
    """
    Returns a dolfin.Function `f` (which by default is defined on the cells)
    such that the value of `f` on any cell whose region number (as stored in
    the mesh file, e.g. when produced by Netgen) is equal to `region_no`.
    The value of `f` for all other cells will be set to `default_value`.

    The returned function will be defined on the cells unless project_to_CG
    has been set to True (then it will be defined on the nodes).

    """
    DG0 = df.FunctionSpace(mesh, "DG", 0)
    f = df.Function(DG0)

    regions = mesh.domains().cell_domains()
    for cell_no, region_no in enumerate(regions.array()):
        # this assumes that the dofs are ordered like the regions information
        if region_no == region:
            f.vector()[cell_no] = value
        else:
            f.vector()[cell_no] = default_value

    if project_to_CG == True:
        return df.project(f, df.FunctionSpace(mesh, "CG", 1))
    return f


def mark_subdomain_by_function(fun,mesh_or_space,domain_index,subdomains):
    """
    Mark the subdomains with given index if user provide a region by function, such as

    def region1(coords):
        if coords[2]<0.5:
            return 1
        else:
            return 0

    """
    if isinstance(mesh_or_space, df.FunctionSpace):
        dg = mesh_or_space
        mesh = dg.mesh()
    else:
        mesh = mesh_or_space

    if hasattr(fun, '__call__'):
        cds=mesh.coordinates()

        index=0
        for cell in df.cells(mesh):
            p1,p2,p3,p4=cell.entities(0)
            coord=(cds[p1]+cds[p2]+cds[p3]+cds[p4])/4.0
            if fun(coord):
                subdomains.array()[index] = domain_index
            index+=1

    else:
        raise AttributeError


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

       filename -- if given, save the resulting plot to a file with the specified name;
                   can also be a list of files
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
        filenames = [filename] if isinstance(filename, basestring) else filename

        for name in filenames:
            create_missing_directory_components(name)
            fig.savefig(name)

def duplicate_output_to_file(filename, add_timestamp=False, timestamp_fmt='__%Y-%m-%d_%H.%M.%S'):
    """
    Redirect all (future) output to a file with the given filename.
    This redirects both to stdout and stderr.

    If `add_timestamp` is True (default: False) then a timestamp will
    be added to the filename indicating the time when the call to this
    function occurred (for example, the filename 'output.txt' might be
    changed into 'output_2012-01-01_14.33.52.txt'). The timestamp
    format can be controlled via `timestamp_fmt`, which should be a
    formatting string as accepted by `datetime.strftime`.
    """
    create_missing_directory_components(filename)
    if add_timestamp:
        name, ext = os.path.splitext(filename)
        filename = '{}{}{}'.format(name, datetime.strftime(datetime.now(), timestamp_fmt), ext)

    logger.debug("Duplicating output to file '{}'".format(filename))
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    tee = sp.Popen(["tee", filename], stdin=sp.PIPE)
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

    a = np.asarray(mesh_function.array() - 1, dtype=np.int32)
    f.vector()[:] = np.choose(a, fun_vals)
    return f


def vector_field_from_dolfin_function(f, xlims=None, ylims=None, zlims=None,
                                      nx=None, ny=None, nz=None):
    """
    Probes a (vector-valued) dolfin.Function `f`: R^3 -> R^3 on a
    rectangular grid.

    It returns six arrays `x`, `y`, `z`, `u`, `v`, `w` representing
    the grid coordinates and vector field components, respectively,
    which can be used for plotting, etc.

    The arguments `xlims`, `ylims`, `zlims`, `nx`, `ny`, `nz` can be
    used to control the extent and coarseness of the grid.


    *Arguments*

    xlims, ylims, zlims : pair of floats

        The extent of the grid along the x-/y-/z-axis. If no value is
        provided, the minimum/maximum mesh coordinates are used along
        each axis.

    nx, ny, nz : int

        The grid spacings along the x-/y-/z/axis. If no value is
        provided, a sensible default is derived from the average cell
        size of the mesh.


    *Returns*

    The arrays `x`, `y`, `z`, `u`, `v`, `w`. Each of these has shape
    (nx, ny, nz). The first three are the same as would be returned by
    the command:

       numpy.mgrid[xmin:xmax:nx*1j, ymin:ymax:ny*1j, zmin:zmax:nz*1j]

    """
    mesh = f.function_space().mesh()
    coords = mesh.coordinates()

    def _find_limits(limits, i):
        return limits if (limits != None) else (min(coords[:, i]), max(coords[:, i]))

    (xmin, xmax) = _find_limits(xlims, 0)
    (ymin, ymax) = _find_limits(ylims, 1)
    (zmin, zmax) = _find_limits(zlims, 2)

    print "Limits:"
    print "xmin, xmax: {}, {}".format(xmin, xmax)
    print "ymin, ymax: {}, {}".format(ymin, ymax)
    print "zmin, zmax: {}, {}".format(zmin, zmax)

    if nx == None or ny == None or nz == None:
        raise NotImplementedError("Please provide specific values "
                                  "for nx, ny, nz for now.")

    X, Y, Z = np.mgrid[xmin:xmax:nx*1j, ymin:ymax:ny*1j, zmin:zmax:nz*1j]

    U = np.empty((nx, ny, nz))
    V = np.empty((nx, ny, nz))
    W = np.empty((nx, ny, nz))

    for i in xrange(nx):
        for j in xrange(ny):
            for k in xrange(nz):
                val = f(X[i,j,k], Y[i, j, k], Z[i, j, k])
                U[i, j, k] = val[0]
                V[i, j, k] = val[1]
                W[i, j, k] = val[2]

    return X, Y, Z, U, V, W


def probe(dolfin_function, points, apply_func=None):
    """
    Probe the dolfin function at the given points.

    *Arguments*

    dolfin_function: dolfin.Function

        A dolfin function.

    points: numpy.array

        An array of points where the field should be probed. Can
        have arbitrary shape, except that the last axis must have
        dimension 3. For example, if pts.shape == (10,20,5,3) then
        the field is probed at all points on a regular grid of
        size 10 x 20 x 5.

    apply_func: any callable

        Optional function to be applied to the returned values. If not
        provided, the values are returned unchanged.

    *Returns*

    A numpy.ma.masked_array of the same shape as `pts`, where the last
    axis contains the field values instead of the point locations (or
    `apply_func` applied to the field values in case it is provided.).
    Positions in the output array corresponding to probing point which
    lie outside the mesh are masked out.

    """
    points = np.array(points)
    if not points.shape[-1] == 3:
        raise ValueError(
            "Arguments 'points' must be a numpy array of 3D points, "
            "i.e. the last axis must have dimension 3. Shape of "
            "'pts' is: {}".format(points.shape))

    if apply_func == None:
        # use the identity operation by default
        apply_func = lambda x: x

    output_shape = np.array(apply_func([0, 0, 0])).shape

    res = np.ma.empty(points.shape[:-1] + output_shape)
    # N.B.: setting the mask to a full matrix right from the start (as
    # we do in the next line) might be slightly memory-inefficient; if
    # that becomes a problem we can always set it to 'np.ma.nomask'
    # here, but then we need a check for res.mask == np.ma.nomask in
    # the 'except' branch below and set it to a full mask if we
    # actually need to mask out any values during the loop.
    res.mask = np.zeros_like(res, dtype=bool)
    loop_indices = itertools.product(*map(xrange, points.shape[:-1]))
    for idx in loop_indices:
        try:
            # XXX TODO: The docstring of a df.Function says at the very
            # end that it's possible to pass (slices of) a larger array
            # in order to fast fill up an array with multiple evaluations.
            # This might be worth investigating!
            #
            # Alternatively, it may be good to write special helper functions
            # For the most common cases Nx3 and (nx x ny x nz x 3). Or can
            # we even reshape the array in the beginning and then only use the
            # first case?
            pt = points[idx]
            res[idx] = apply_func(dolfin_function(pt))
        except RuntimeError:
            res.mask[idx] = True

    return res


def compute_dmdt(t0, m0, t1, m1):
    """
    Returns the maximum of the L2 norm of dm/dt.

    Arguments:
        t0, t1: two points in time (floats)
        m0, m1: the magnetisation at t0, resp. t1 (np.arrays of shape 3*n)

    """
    dm = (m1 - m0).reshape((3, -1))
    max_dm = np.max(np.sqrt(np.sum(dm**2, axis=0))) # max of L2-norm
    dt = abs(t1 - t0)
    max_dmdt = max_dm / dt
    return max_dmdt


def npy_to_dolfin_function(filename, mesh_or_space):
    """
    Create a dolfin function on the given mesh or function space whose
    coefficients are stored in the given file.

    *Arguments*

    filename : str

       The .npy file in which the function coefficients are stored.

    mesh_or_space :

        Either a dolfin.Mesh or a dolfin.VectorFunctionSpace of dimension 3.

    """
    # XXX TODO: Implement a test which writes a function, then reads
    # it back in and checks that it's the same. Also vice versa.
    a = np.load(filename)
    _, V = mesh_and_space(mesh_or_space)
    fun = df.Function(V)
    fun.vector().set_local(a)
    return fun


def average_field(field_vals):
    """
    Return the average value of the given field. `field_vals` must be
    a numpy.array of shape (N,) followingg the dolfin convention for
    the field values.
    """
    assert field_vals.ndim == 1

    field_vals.shape = (3,-1)
    av = np.average(field_vals, axis=1)
    field_vals.shape = (-1,)
    return av


def save_dg_fun(fun, name='unnamed.vtk', dataname='m', binary=False):
    """
    Seems that saving DG function to vtk doesn't work properly.
    Ooops, seems that paraview don't like cell data.
    """
    import pyvtk

    funspace = fun.function_space()
    mesh = funspace.mesh()

    points = mesh.coordinates()
    tetras = np.array(mesh.cells(),dtype=np.int)

    grid = pyvtk.UnstructuredGrid(points,
                                  tetra = tetras)

    m = fun.vector().array()

    m.shape=(3,-1)
    print m
    data=pyvtk.CellData(pyvtk.Vectors(np.transpose(m)))
    m.shape=(-1,)

    vtk = pyvtk.VtkData(grid,data,dataname)

    if binary:
        vtk.tofile(name,'binary')
    else:
        vtk.tofile(name)


def save_dg_fun_points(fun, name='unnamed.vtk', dataname='m', binary=False):
    """
    Seems that saving DG function to vtk doesn't work properly.
    """
    import pyvtk

    V = fun.function_space()
    mesh = V.mesh()

    points = []
    for cell in df.cells(mesh):
        points.append(V.dofmap().tabulate_coordinates(cell)[0])


    verts = [i for i in range(len(points))]

    grid = pyvtk.UnstructuredGrid(points,
                                  vertex=verts)

    m = fun.vector().array()

    m.shape=(3,-1)
    data=pyvtk.PointData(pyvtk.Vectors(np.transpose(m),dataname))
    m.shape=(-1,)


    vtk = pyvtk.VtkData(grid,data, 'Generated by Finmag')

    if binary:
        vtk.tofile(name,'binary')
    else:
        vtk.tofile(name)


def plot_ndt_columns(ndt_file, columns=['m_x', 'm_y', 'm_z'], style='-',
                     xlim=None, ylim=None, outfile=None, title=None,
                     show_legend=True, legend_loc='best', figsize=None):
    """
    Helper function to quickly plot the time evolution of the specified
    columns in an .ndt file (default: m_x, m_y, m_z) and optionally save
    the output to a file.

    *Arguments*

    columns :  list of strings

        The names of the columns to plot. These must coincide with the
        names in the first header line of the .ndt file.

    outfile :  None | string

        If given, save the plot to a file with this name.

    style :  string

        The plotting style. Default: '-'.

    title :  None | string

        Title text to use for the plot.

    show_legend :  boolean

        If True, a legend with the column names is displayed.

    legend_loc :  string | integer

        Optional location code for the legend (same as for pyplot's
        legend() command).

    figsize :  None | pair of float

        Optional argument to set the figure size of the output plot.
    """
    f = Tablereader(ndt_file)
    ts = f.timesteps()
    column_vals = f[tuple(columns)]
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    for col_name, col in zip(columns, column_vals):
        ax.plot(ts, col, style, label=col_name)
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc=legend_loc)

    ymin, ymax = ylim or ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ymin - 0.05, ymax + 0.05)

    if outfile:
        fig.savefig(outfile)
    return fig


def plot_dynamics(ndt_file, components='xyz', **kwargs):
    """
    Convenience wrapper around `plot_ndt_columns` with a slightly
    easier to remember name. The main difference is that this function
    can only plot the dynamics of the magnetisation, not other fields.

    The magnetisation components to plot are specified in the argument
    `components`. All other keyword arguments are the same as for
    `plot_ndt_columns`.

    """
    columns = ['m_{}'.format(c) for c in components]
    return plot_ndt_columns(ndt_file, columns=columns, **kwargs)


def vec2str(a, fmt='{}', delims='()', sep=', '):
    """
    Convert a 3-sequence (e.g. a numpy array) to a string, optionally
    with some formatting options. The argument `a` is also allowed to
    have the value `None`, in which case the string 'None' is returned.

    The argument `delims` can be used to specify different left and right
    delimiters (default: opening and closing parentheses). If only one
    delimiter is given (e.g. "|") then this is used both as left and right
    delimiter. If `delims` is empty, no delimiters will be used.

    Examples::

       a = numpy.array([1, 200, 30000])
       vec2str(a)  -->  (1, 200, 30000)
       vec2str(a, fmt='{:.3g}')  -->  (1, 200, 3e+04)
       vec2str(a, fmt='{:.2f}')  -->  (1.00, 200.00, 30000.00)
       vec2str(a, delims='[]')  -->  [1, 200, 30000]
       vec2str(a, delims='|', sep='__')  -->  |1__200__30000|
       vec2str(a, delims='', sep=' - ')  -->  1 - 200 - 30000

    """
    if a is None:
        res = 'None'
    else:
       try:
           ldelim = delims[0]
       except IndexError:
           ldelim = ""
       try:
           rdelim = delims[1]
       except IndexError:
           rdelim = ldelim
       res = ("{ldelim}{fmt}{sep}{fmt}{sep}{fmt}{rdelim}".format(fmt=fmt, ldelim=ldelim, rdelim=rdelim, sep=sep)).format(a[0], a[1], a[2])
    return res


def pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


def apply_vertexwise(f, *args):
    """
    Apply the function f to the values of the given arguments
    at each vertex separately and aggregate the result into a
    new dolfin.Function.

    *Arguments*

    f :  callable

        The function that is to be applied to the vertex values.

    \*args:  collection of dolfin.Functions

        The fields

    """
    # XXX TODO: Some of this is horribly inefficient. Should clean it up...
    mesh = args[0].function_space().mesh()
    # XXX TODO: the following check(s) seems to fail even if the meshes are the same. How to do this properly in dolfin?
    #assert(all((mesh == u.function_space().mesh() for u in args)))  # check that all meshes coincide
    #assert(all(u.function_space().mesh() == u.function_space().mesh() for (u, v) in pairwise(args)))  # check that all meshes coincide

    # Extract the array for each dolfin.Function
    aa = [u.vector().array() for u in args]
    #V = args[0].function_space()
    #assert(all([V == a.function_space() for a in aa]))  # XXX TODO: how to deal with functions defined on different function spaces?!?

    # Reshape each array according to the dimension of the VectorFunctionSpace
    dims = [u.domain().geometric_dimension() for u in args]
    aa_reshaped = [a.reshape(dim, -1).T for (a, dim) in itertools.izip(aa, dims)]

    # Evaluate f on successive rows of the reshaped arrays
    aa_evaluated = [f(*args) for args in itertools.izip(*aa_reshaped)]

    #import ipdb; ipdb.set_trace()
    try:
        dim_out = len(aa_evaluated[0])
    except TypeError:
        dim_out = None

    if dim_out is None:
        W = df.FunctionSpace(mesh, 'CG', 1)
    else:
        assert(all(dim_out == len(x) for x in aa_evaluated))
        W = df.VectorFunctionSpace(mesh, 'CG', 1, dim=dim_out)  # XXX TODO: should we use df.FunctionSpace if dim_out == 1 ?
    res = df.Function(W)
    res.vector().set_local(np.array(aa_evaluated).T.reshape(-1))
    return res


class TemporaryDirectory(object):
    def __init__(self, keep=False):
        self.keep = keep

    def __enter__(self):
        self.tmpdir = tempfile.mkdtemp()
        return self.tmpdir

    def __exit__(self, type, value, traceback):
        if not self.keep:
            shutil.rmtree(self.tmpdir)
            self.tmpdir = None


@contextmanager
def ignored(*exceptions):
    """
    Ignore the given exceptions within the scope of the context.

    Example::

       with ignored(OSError):
           os.remove('non_existing_file.txt')

    """
    try:
        yield
    except exceptions:
        pass


def run_cmd_with_timeout(cmd, timeout_sec):
    """
    Runs the given shell command but kills the spawned subprocess
    if the timeout is reached.

    Returns the exit code of the shell command. Raises OSError if
    the command does not exist. If the timeout is reached and the
    process is killed, the return code is -9.

    """
    proc = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.PIPE)
    kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    timer.start()
    stdout, stderr = proc.communicate()
    timer.cancel()
    return proc.returncode, stdout, stderr


def jpg2avi(jpg_filename, outfilename=None, duration=1, fps=25):
    """
    Convert a series of .jpg files into an animation file in .avi format.

    *Arguments*

    jpg_filename:

        The 'basename' of the series of image files. For example, if
        the image files are called 'foo_0000.jpg', 'foo_0001.jpg', etc.
        then `jpg_filename` should be 'foo.jpg'. Internally, the image
        files are found via a wildcard expression by replacing the suffix
        ``.jpg`` with ``*.jpg``, so the basename could also have been
        ``foo_.jpg`` or even ``f.jpg``. (However, it should be restrictive
        enough so that only the desired images are found.)

    outfilename:

        The filename of the resulting .avi file. If None (the default),
        uses the basename of the .jpg file.

    duration:

        Duration of the created animation (in seconds).

    fps:

        Framerate (in frames per second) of the created animation.

    """
    if not jpg_filename.endswith('.jpg'):
        raise ValueError("jpg_filename must end in '.jpg'. Got: '{}'".format(jpg_filename))

    pattern = re.sub('\.jpg$', '*.jpg', jpg_filename)
    pattern_escaped = re.sub('\.jpg$', '\*.jpg', jpg_filename)
    jpg_files = sorted(glob(pattern))
    logger.debug('Found {} jpg files.'.format(len(jpg_files)))

    if outfilename is None:
        outfilename = re.sub('\.jpg$', '.avi', jpg_filename)
    logger.debug("Using outfilename='{}'".format(outfilename))
    create_missing_directory_components(outfilename)

    # Use mencoder with two-pass encoding to convert the files.
    # See http://mariovalle.name/mencoder/mencoder.html
    try:
        mencoder_options = "vbitrate=2160000:mbd=2:keyint=132:v4mv:vqmin=3:lumi_mask=0.07:dark_mask=0.2:mpeg_quant:scplx_mask=0.1:tcplx_mask=0.1:naq"
        sh.mencoder('-ovc', 'lavc', '-lavcopts',
                    'vcodec=mpeg4:vpass=1:' + mencoder_options,
                    '-mf', 'type=jpg:fps={}'.format(fps/duration), '-nosound',
                    '-o', '/dev/null', 'mf://' + pattern)
        sh.mencoder('-ovc', 'lavc', '-lavcopts',
                    'vcodec=mpeg4:vpass=2:' + mencoder_options,
                    '-mf', 'type=jpg:fps={}'.format(fps/duration), '-nosound',
                    '-o', outfilename, 'mf://' + pattern)
        os.remove('divx2pass.log')  # tidy up output from the two-pass enoding
    except sh.CommandNotFound:
        logger.error("mencoder does not seem to be installed but is needed for "
                  "movie creation. Please install it (e.g. on Debian/Ubuntu: "
                  "'sudo apt-get install mencoder').")
    except sh.ErrorReturnCode as exc:
        logger.warning("mencoder had non-zero exit status: {} (diagnostic message: '{}')".format(exc.exit_code, exc.message))



def pvd2avi(pvd_filename, outfilename=None, duration=1, fps=25, **kwargs):
    """
    Export a .pvd animation to a movie file in .avi format.

    *Arguments*

    pvd_filename:

        The name of the .pvd file to be converted.

    outfilename:

        The filename of the resulting .avi file. If None (the default),
        the basename of the .pvd file is used.

    duration:

        Duration of the created animation (in seconds).

    fps:

        Framerate (in frames per second) of the created animation.

    All other keyword arguments are passed on to the function
    `finmag.util.visualization.render_paraview_scene` and can
    be used to tweak the appearance of the animation.

    """
    if not pvd_filename.endswith('.pvd'):
        raise ValueError("pvd_filename must end in '.pvd'. Got: '{}'".format(pvd_filename))

    if outfilename == None:
        outfilename = re.sub('\.pvd$', '.avi', pvd_filename)

    if kwargs.has_key('trim_border') and kwargs['trim_border'] ==True:
        logger.warning("Cannot use 'trim_border=True' when converting a .pvd time series to .avi; using 'trim_border=False'.")
    kwargs['trim_border'] = False

    with TemporaryDirectory() as tmpdir:
        jpg_tmpfilename = os.path.join(tmpdir, 'animation.jpg')
        render_paraview_scene(pvd_filename, outfile=jpg_tmpfilename, **kwargs)
        jpg2avi(jpg_tmpfilename, outfilename=outfilename, duration=duration, fps=fps)


def warn_about_outdated_code(min_dolfin_version, msg):
    """
    If the current dolfin version is >= min_dolfin_version, print the
    given warning message. This is useful to warn about outdated code
    which is temporarily kept for backwards compatibility but should
    eventually be removed. Remember that the call to this function
    should of course occur from the new code that will be executed in
    later dolfin versions (otherwise the warning will never get
    printed when it's relevant).

    """
    if LooseVersion(get_version_dolfin()) >= LooseVersion(min_dolfin_version):
        logger.warning(msg)
