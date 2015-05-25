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


def expression_from_python_function(func, function_space):
    class ExpressionFromPythonFunction(df.Expression):
        """
        Turn a python function to a dolfin expression over given functionspace.

        """
        def __init__(self, python_function):
            self.func = python_function

        def eval(self, eval_result, x):
            eval_result[:] = self.func(x)

        def value_shape(self):
            # () for scalar field, (N,) for N dimensional vector field
            return function_space.ufl_element().value_shape()
    return ExpressionFromPythonFunction(func)


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
        # check that we have any handlers at all before we attempt
        # to iterate
        if hasattr(logger, 'handlers'):
            for i, handler in enumerate(logger.handlers):
                handlerstr = logging_handler_str(handler)
                msg += (" %15s (lev=%2d, eff.lev=%2d) -> handler %d: lev=%2d %s\n"
                        % (loggername, logger.level, logger.getEffectiveLevel(),
                           i, handler.level, handlerstr))
        else:
                msg += (" %15s -> %s\n"
                        % (loggername, "no handlers found"))

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
supported_color_schemes_str = ", ".join(
    ["'{}'".format(s) for s in supported_color_schemes])


def set_color_scheme(color_scheme):
    """
    Set the color scheme for finmag log messages in the terminal.

    *Arguments*

    color_scheme: string

       One of the color schemes supported by Python's `logging` module.
       Supported values: {}.

    """
    if color_scheme not in supported_color_schemes:
        raise ValueError(
            "Color scheme must be one of: {}".format(supported_color_schemes_str))
    for h in logger.handlers:
        if not isinstance(h, ansistrm.ColorizingStreamHandler):
            continue
        h.level_map = ansistrm.level_maps[color_scheme]
# Insert supported color schemes into docstring
set_color_scheme.__doc__ = set_color_scheme.__doc__.format(
    supported_color_schemes_str)


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
    h = logging.handlers.RotatingFileHandler(
        filename, mode=mode, maxBytes=maxBytes, backupCount=backupCount)
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
        raise ValueError(
            "Expected a valid repository, but directory does not exist: '{}'".format(repo_dir))

    try:
        rev_nr = int(
            sp.check_output(['hg', 'id', '-n', '-r', revision]).strip())
        rev_id = sp.check_output(['hg', 'id', '-i', '-r', revision]).strip()
        #rev_log = sp.check_output(['hg', 'log', '-r', revision]).strip()
        rev_date = sp.check_output(
            ['hg', 'log', '-r', revision, '--template', '{date|isodate}']).split()[0]
    except sp.CalledProcessError:
        raise ValueError(
            "Invalid revision '{}', or invalid Mercurial repository: '{}'".format(revision, repo_dir))

    os.chdir(cwd_bak)
    return rev_nr, rev_id, rev_date

def get_git_revision_info(repo_dir, revision='HEAD'):
    """
    Return the revision id and the date of a revision in the given github repository.
    For examaple, the information returned should looks like,

    """
    cwd_bak = os.getcwd()
    try:
        os.chdir(os.path.expanduser(repo_dir))
    except OSError:
        raise ValueError(
            "Expected a valid repository, but directory does not exist: '{}'".format(repo_dir))

    try:
        rev_id = sp.check_output(['git', 'rev-parse',  revision]).strip()
        rev_date = sp.check_output(
            ['git', 'show', '-s', '--format=%ci',revision]).split()[0]

    except sp.CalledProcessError:
        raise ValueError(
            "Invalid revision '{}', or invalid Git repository: '{}'".format(revision, repo_dir))

    os.chdir(cwd_bak)
    return rev_id, rev_date

def binary_tarball_name(repo_dir, revision='HEAD', suffix=''):
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
    # XXX TODO: Should we also check whether the repo is actually a Finmag
    # repository?!?
    rev_id, rev_date = get_git_revision_info(repo_dir, revision)
    tarball_name = "FinMag-dist__{}__{}{}.tar.bz2".format(
        rev_date,  rev_id, suffix)
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


def times_curl(v, dim):
    """
    Returns v times curl of v on meshes with dimensions 1, 2, or 3.

    Arguments:

        - v is a dolfin function on a 1d, 2d or 3d vector function space.
        - dim is the number of dimensions of the mesh.

    On three-dimensional meshes, dolfin supports computing the integrand
    of the DMI energy using

         df.inner(v, df.curl(v))        eq.1

    However, the curl operator is not implemented on 1d and 2d meshes.
    With the expansion of the curl in cartesian coordinates:

        curlx = dmzdy - dmydz
        curly = dmxdz - dmzdx
        curlz = dmydx - dmxdy

    we can compute eq. 1 with

        (vx * curlx + vy * curly + vz * curlz),

    including only existing derivatives. Derivatives that do not exist
    are set to 0.

    """
    if dim == 3:
        return df.inner(v, df.curl(v))

    gradv = df.grad(v)

    # Derivatives along x direction exist in both 1d and 2d cases.
    dvxdx = gradv[0, 0]
    dvydx = gradv[1, 0]
    dvzdx = gradv[2, 0]

    # Derivatives along z direction do not exist in 1d and 2d cases,
    # so they are set to zero.
    dvxdz = 0
    dvydz = 0
    dvzdz = 0

    # Derivatives along y direction exist only in a 2d case.
    # For 1d case, these derivatives are set to zero.
    if dim == 1:
        dvxdy = 0
        dvydy = 0
        dvzdy = 0
    elif dim == 2:
        dvxdy = gradv[0, 1]
        dvydy = gradv[1, 1]
        dvzdy = gradv[2, 1]

    # Components of the curl(v).
    curlx = dvzdy - dvydz
    curly = dvxdz - dvzdx
    curlz = dvydx - dvxdy

    # Return v*curl(v).
    return v[0] * curlx + v[1] * curly + v[2] * curlz


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
    number_of_nodes = len(vs) / 3
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
    return np.sqrt(np.add.reduce(vs * vs, axis=1))


def crossprod(v, w):
    """
    Compute the point-wise cross product of two (3D) vector fields on a mesh.

    The arguments `v` and `w` should be numpy.arrays representing
    dolfin functions, i.e. they should be of the form [x0, ..., xn,
    y0, ..., yn, z0, ..., zn]. The return value is again a numpy array
    of the same form.

    """
    if df.parameters.reorder_dofs_serial != False:
        raise RuntimeError(
            "Please ensure that df.parameters.reorder_dofs_serial is set to False.")
    assert(v.ndim == 1 and w.ndim == 1)
    a = v.reshape(3, -1)
    b = w.reshape(3, -1)
    return np.cross(a, b, axisa=0, axisb=0, axisc=0).reshape(-1)


def fnormalise(arr, ignore_zero_vectors=False):
    """
    Returns a normalised copy of vectors in arr.

    Expects arr to be a numpy.ndarray in the form that dolfin
    provides: [x0, ..., xn, y0, ..., yn, z0, ..., zn].

    If `ignore_zero_vectors` is True (default: False) then any
    3-vector of norm zero will be left alone. Thus the resulting array
    will still have zero vectors in the same places. Otherwise the
    entries in the resulting array will be filled with NaN.

    """
    a = arr.astype(np.float64)  # this copies

    a = a.reshape((3, -1))
    a_norm = np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    if ignore_zero_vectors:
        a_norm[np.where(a_norm == 0)] = 1.0
    a /= a_norm
    a = a.ravel()
    return a


def angle(v1, v2):
    """
    Returns the angle between two three-dimensional vectors.

    """
    return np.arccos(np.dot(v1, v2) / (norm(v1) * norm(v2)))


def rows_to_columns(arr):
    """
    For an array of the shape [[x1, y1, z1], ..., [xn, yn, zn]]
    returns an array of the shape [[x1, ..., xn],[y1, ..., yn],[z1, ..., zn]].

    """
    return arr.reshape(arr.size, order="F").reshape((3, -1))


def stats(arr, axis=1):
    median = np.median(arr)
    average = np.mean(arr)
    minimum = np.nanmin(arr)
    maximum = np.nanmax(arr)
    spread = np.std(arr)
    stats = "    min, median, max = {0}, {1}, {2}\n    mean, std = {3}, {4}".format(
        minimum, median, maximum, average, spread)
    return stats


def frexp10(x, method="string"):
    """
    Same as math.frexp but in base 10, will return (m, e)
    such that x == m * 10 ** e.

    """
    if method == "math":
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


def verify_function_space_type(function_space, family, degree, dim):
    """
    Check that `function_space` is a dolfin FunctionSpace or VectorFunctionSpace
    of the correct type. It checks for:

       - the finite element family (e.g. 'CG' for a CG1 space)

       - the finite element degree (e.g. 1 for a CG1 space

       - the dimension `dim` of the function values of elements of that function space;
         this would be `None` if `function_space` is a FunctionSpace and a single number
         if `function_space` is a VectorFunctionSpace.

    """
    ufl_element = function_space.ufl_element()
    mesh = function_space.mesh()

    # Allow abbreviations 'DG' and 'CG'
    if family == 'DG':
        family = 'Discontinuous Lagrange'
    elif family == 'CG':
        family = 'Lagrange'

    family_and_degree_are_correct = \
        (family == ufl_element.family() and
         degree == ufl_element.degree())

    if dim == None:
        # `function_space` should be a dolfin.FunctionSpace
        return (isinstance(function_space, df.FunctionSpace) and
                family_and_degree_are_correct)
    else:
        # `function_space` should be a dolfin.VectorFunctionSpace
        # for VectorFunctionSpace this should be a 1-tuple of the form (dim,)
        value_shape = ufl_element.value_shape()
        if len(value_shape) != 1:
            return False
        else:
            return (isinstance(function_space, df.VectorFunctionSpace) and
                    family_and_degree_are_correct and
                    dim == value_shape[0])


def mesh_equal(mesh1, mesh2):
    cds1 = mesh1.coordinates()
    cds2 = mesh2.coordinates()
    return np.array_equal(cds1, cds2)

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
    assert(S3.ufl_element().family() == 'Lagrange')
    assert(S3.ufl_element().degree() == 1)

    if isinstance(value, (df.Constant, df.Expression)):
        fun = df.interpolate(value, S3)
    elif isinstance(value, (tuple, list, np.ndarray)) and len(value) == 3:
        # We recognise a sequence of strings as ingredient for a df.Expression.
        if all(isinstance(item, basestring) for item in value):
            expr = df.Expression(value, **kwargs)
            fun = df.interpolate(expr, S3)
        else:
            #fun = df.Function(S3)
            #vec = np.empty((fun.vector().size()/3, 3))
            # vec[:] = value # using broadcasting
            # fun.vector().set_local(vec.transpose().reshape(-1))
            expr = df.Constant(list(value))
            fun = df.interpolate(expr, S3)

    elif isinstance(value, np.ndarray):
        fun = df.Function(S3)
        if value.ndim == 2:
            assert value.shape[1] == 3
            value = value.reshape(value.size, order="F")
        if not value.dtype == np.double:
            value = value.astype(np.double)
        fun.vector().set_local(value)

    # if it's a normal function, we wrapper it into a dolfin expression
    elif hasattr(value, '__call__'):

        class HelperExpression(df.Expression):

            def __init__(self, value):
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
        fun.vector().set_local(fnormalise(fun.vector().array()))

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
        assert(S1.ufl_element().family() == 'Lagrange')
        assert(S1.ufl_element().degree() == 1)
        mesh = S1.mesh()
    else:
        mesh = mesh_or_space
        S1 = df.FunctionSpace(mesh, "Lagrange", 1)

    if isinstance(value, (df.Constant, df.Expression)):
        fun = df.interpolate(value, S1)
    elif isinstance(value, (np.ndarray, list)):
        fun = df.Function(S1)
        assert(len(value) == fun.vector().size())
        fun.vector().set_local(value)
    elif isinstance(value, (int, float, long)):
        fun = df.Function(S1)
        fun.vector()[:] = value
    elif hasattr(value, '__call__'):

        # if it's a normal function, we wrapper it into a dolfin expression
        class HelperExpression(df.Expression):

            def __init__(self, value):
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
        assert(dg.ufl_element().family() == 'Discontinuous Lagrange')
        assert(dg.ufl_element().degree() == 0)
        mesh = dg.mesh()
    else:
        mesh = mesh_or_space
        dg = df.FunctionSpace(mesh, "DG", 0)

    if isinstance(value, (df.Constant, df.Expression)):
        fun = df.interpolate(value, dg)
    elif isinstance(value, (np.ndarray, list)):
        fun = df.Function(dg)
        assert(len(value) == fun.vector().size())
        fun.vector().set_local(value)
    elif isinstance(value, (int, float, long)):
        fun = df.Function(dg)
        fun.vector()[:] = value
    elif isinstance(value, df.Function):
        mesh1 = value.function_space().mesh()
        fun = df.Function(dg)
        if mesh_equal(mesh, mesh1) and value.vector().size() == fun.vector().size():
            fun = value
        else:
            raise RuntimeError("Meshes are not compatible for given function.")
    elif hasattr(value, '__call__'):

        class HelperExpression(df.Expression):

            def __init__(self, value):
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
            vec = np.empty((fun.vector().size() / 3, 3))
            vec[:] = value  # using broadcasting

            fun.vector().set_local(vec.transpose().reshape(-1))
    elif isinstance(value, (np.ndarray, list)):
        fun = df.Function(dg)
        assert(len(value) == fun.vector().size())
        fun.vector().set_local(value)
    elif isinstance(value, (int, float, long)):
        fun = df.Function(dg)
        fun.vector()[:] = value
    elif isinstance(value, df.Function):
        mesh1 = value.function_space().mesh()
        fun = df.Function(dg)
        if mesh_equal(mesh, mesh1) and value.vector().size() == fun.vector().size():
            fun = value
        else:
            raise RuntimeError("Meshes are not compatible for given function.")
    elif hasattr(value, '__call__'):

        class HelperExpression(df.Expression):

            def __init__(self, value):
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


def restriction(mesh, submesh):
    """
    Return a Python function `r` of the form

       r: f -> f_submesh

    whose first argument `f` is either a `dolfin.Function` or a
    `numpy.array` and which returns another `dolfin.Function` or
    `numpy.array` which has the same values as `f` but is only
    defined on `submesh`.

    `submesh` must be of type `dolfin.SubMesh` and be a proper
    submesh of `mesh`.
    """
    if not isinstance(submesh, df.SubMesh):
        raise TypeError("Argument 'submesh' must be of type `dolfin.SubMesh`. "
                        "Got: {}".format(type(submesh)))

    try:
        # This is the correct syntax now, see:
        # http://fenicsproject.org/qa/185/entity-mapping-between-a-submesh-and-the-parent-mesh
        parent_vertex_indices = submesh.data().array(
            'parent_vertex_indices', 0)
    except RuntimeError:
        # Legacy syntax (for dolfin <= 1.2 or so).
        # TODO: This should be removed in the future once dolfin 1.3 is
        # released!
        parent_vertex_indices = submesh.data().mesh_function(
            'parent_vertex_indices').array()

    V = df.FunctionSpace(mesh, 'CG', 1)
    V_submesh = df.FunctionSpace(submesh, 'CG', 1)

    def restrict_to_submesh(f):
        # Remark: We can't use df.interpolate here to interpolate the
        # function values from the full mesh on the submesh because it
        # sometimes crashes (probably due to rounding errors), even if we
        # set df.parameters["allow_extrapolation"]=True as they recommend
        # in the error message.
        #
        # Therefore we manually interpolate the function values here using
        # the vertex mappings determined above. This works fine if the
        # dofs are not re-ordered, but will probably cause problems in
        # parallel (or with dof reordering enabled).
        if isinstance(f, np.ndarray):
            if f.ndim == 1:
                return f[parent_vertex_indices]
            elif f.ndim == 2:
                return f[:, parent_vertex_indices]
            else:
                raise TypeError(
                    "Array must be 1- or 2-dimensional. Got: dim={}".format(f.ndim))
        else:
            assert(isinstance(f, df.Function))
            f_arr = f.vector().array()
            f_submesh = df.Function(V_submesh)
            f_submesh.vector()[:] = f_arr[parent_vertex_indices]
            return f_submesh

    return restrict_to_submesh


def mark_subdomain_by_function(fun, mesh_or_space, domain_index, subdomains):
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
        cds = mesh.coordinates()

        index = 0
        for cell in df.cells(mesh):
            p1, p2, p3, p4 = cell.entities(0)
            coord = (cds[p1] + cds[p2] + cds[p3] + cds[p4]) / 4.0
            if fun(coord):
                subdomains.array()[index] = domain_index
            index += 1

    else:
        raise AttributeError


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
        filename = '{}{}{}'.format(
            name, datetime.strftime(datetime.now(), timestamp_fmt), ext)

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

    X, Y, Z = np.mgrid[xmin:xmax:nx * 1j, ymin:ymax:ny * 1j, zmin:zmax:nz * 1j]

    U = np.empty((nx, ny, nz))
    V = np.empty((nx, ny, nz))
    W = np.empty((nx, ny, nz))

    for i in xrange(nx):
        for j in xrange(ny):
            for k in xrange(nz):
                val = f(X[i, j, k], Y[i, j, k], Z[i, j, k])
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
    Elements in the output array corresponding to probing point outside
    the mesh are masked out.

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


def probe_along_line(dolfin_function, pt_start, pt_end, N, apply_func=None):
    """
    Probe the dolfin function at the `N` equidistant points along a straight
    line connecting `pt_start` and `pt_end`.

    *Arguments*

    dolfin_function: dolfin.Function

        A dolfin function.

    pt_start, pt_end:

       Start and end point of the straight line along which to point.

    N: int

       Number of probing points.

    apply_func: any callable

        Optional function to be applied to the returned values. If not
        provided, the values are returned unchanged.

    *Returns*

    A tuple `(pts, vals)` where `pts` is the list of probing points
    (i.e., the `N` equidistant points between `pt_start` and `pt_end`)
    and `vals` is a numpy.ma.masked_array of shape `(N, 3)` containing
    the field values at the probed points (or `apply_func` applied to
    the field values in case it is provided.). Elements in the output
    array corresponding to probing point outside the mesh are masked out.

    """
    pt_start = np.asarray(pt_start)
    pt_end = np.asarray(pt_end)
    pts = np.array(
        [(1 - t) * pt_start + t * pt_end for t in np.linspace(0, 1, N)])
    vals = probe(dolfin_function, pts, apply_func=apply_func)
    return pts, vals


def compute_dmdt(t0, m0, t1, m1):
    """
    Returns the maximum of the L2 norm of dm/dt.

    Arguments:
        t0, t1: two points in time (floats)
        m0, m1: the magnetisation at t0, resp. t1 (np.arrays of shape 3*n)

    """
    dm = (m1 - m0).reshape((3, -1))
    max_dm = np.max(np.sqrt(np.sum(dm ** 2, axis=0)))  # max of L2-norm
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

    field_vals.shape = (3, -1)
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
    tetras = np.array(mesh.cells(), dtype=np.int)

    grid = pyvtk.UnstructuredGrid(points,
                                  tetra=tetras)

    m = fun.vector().array()

    m.shape = (3, -1)
    print m
    data = pyvtk.CellData(pyvtk.Vectors(np.transpose(m)))
    m.shape = (-1,)

    vtk = pyvtk.VtkData(grid, data, dataname)

    if binary:
        vtk.tofile(name, 'binary')
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

    m.shape = (3, -1)
    data = pyvtk.PointData(pyvtk.Vectors(np.transpose(m), dataname))
    m.shape = (-1,)

    vtk = pyvtk.VtkData(grid, data, 'Generated by Finmag')

    if binary:
        vtk.tofile(name, 'binary')
    else:
        vtk.tofile(name)


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
        res = ("{ldelim}{fmt}{sep}{fmt}{sep}{fmt}{rdelim}".format(
            fmt=fmt, ldelim=ldelim, rdelim=rdelim, sep=sep)).format(a[0], a[1], a[2])
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
    # assert(all((mesh == u.function_space().mesh() for u in args)))  # check that all meshes coincide
    # assert(all(u.function_space().mesh() == u.function_space().mesh() for
    # (u, v) in pairwise(args)))  # check that all meshes coincide

    # Extract the array for each dolfin.Function
    aa = [u.vector().array() for u in args]
    #V = args[0].function_space()
    # assert(all([V == a.function_space() for a in aa]))  # XXX TODO: how to
    # deal with functions defined on different function spaces?!?

    # Reshape each array according to the dimension of the VectorFunctionSpace
    dims = [u.domain().geometric_dimension() for u in args]
    aa_reshaped = [
        a.reshape(dim, -1).T for (a, dim) in itertools.izip(aa, dims)]

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
        # XXX TODO: should we use df.FunctionSpace if dim_out == 1 ?
        W = df.VectorFunctionSpace(mesh, 'CG', 1, dim=dim_out)
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


class run_in_tmpdir(object):

    def __init__(self, keep=False):
        self.keep = keep
        self.cwd_bak = os.getcwd()

    def __enter__(self):
        self.tmpdir = tempfile.mkdtemp()
        os.chdir(self.tmpdir)
        return self.tmpdir

    def __exit__(self, type, value, traceback):
        if not self.keep:
            shutil.rmtree(self.tmpdir)
            self.tmpdir = None
        os.chdir(self.cwd_bak)


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
        raise ValueError(
            "jpg_filename must end in '.jpg'. Got: '{}'".format(jpg_filename))

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
                    '-mf', 'type=jpg:fps={}'.format(fps /
                                                    duration), '-nosound',
                    '-o', '/dev/null', 'mf://' + pattern)
        sh.mencoder('-ovc', 'lavc', '-lavcopts',
                    'vcodec=mpeg4:vpass=2:' + mencoder_options,
                    '-mf', 'type=jpg:fps={}'.format(fps /
                                                    duration), '-nosound',
                    '-o', outfilename, 'mf://' + pattern)
        os.remove('divx2pass.log')  # tidy up output from the two-pass enoding
    except sh.CommandNotFound:
        logger.error("mencoder does not seem to be installed but is needed for "
                     "movie creation. Please install it (e.g. on Debian/Ubuntu: "
                     "'sudo apt-get install mencoder').")
    except sh.ErrorReturnCode as exc:
        logger.warning(
            "mencoder had non-zero exit status: {} (diagnostic message: '{}')".format(exc.exit_code, exc.message))


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
        raise ValueError(
            "pvd_filename must end in '.pvd'. Got: '{}'".format(pvd_filename))

    if outfilename == None:
        outfilename = re.sub('\.pvd$', '.avi', pvd_filename)

    if kwargs.has_key('trim_border') and kwargs['trim_border'] == True:
        logger.warning(
            "Cannot use 'trim_border=True' when converting a .pvd time series to .avi; using 'trim_border=False'.")
    kwargs['trim_border'] = False

    with TemporaryDirectory() as tmpdir:
        jpg_tmpfilename = os.path.join(tmpdir, 'animation.jpg')
        render_paraview_scene(pvd_filename, outfile=jpg_tmpfilename, **kwargs)
        jpg2avi(jpg_tmpfilename, outfilename=outfilename,
                duration=duration, fps=fps)


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


def format_time(num_seconds):
    """
    Given a number of seconds, return a string with `num_seconds`
    converted into a more readable format (including minutes and
    hours if appropriate).

    """
    hours = int(num_seconds / 3600.0)
    r = num_seconds - 3600 * hours
    minutes = int(r / 60.0)
    seconds = r - 60 * minutes

    res = "{} h ".format(hours) if (hours > 0) else ""
    res += "{} min ".format(minutes) if (minutes >
                                         0 or (minutes == 0 and hours > 0)) else ""
    res += "{:.2f} seconds".format(seconds)
    return res


def make_human_readable(nbytes):
    """
    Given a number of bytes, return a string of the form "12.2 MB" or "3.44 GB"
    which makes the number more digestible by a human reader. Everything less
    than 500 MB will be displayed in units of MB, everything above in units of GB.
    """
    if nbytes < 500 * 1024 ** 2:
        res = '{:.2f} MB'.format(nbytes / 1024 ** 2)
    else:
        res = '{:.2f} GB'.format(nbytes / 1024 ** 3)
    return res


def print_boundary_element_matrix_size(mesh, generalised=False):
    """
    Given a 3D mesh, print the amount of memory that the boundary element matrix
    will occupy in memory. This is useful when treating very big problems in order
    to "interactively" adjust a mesh until the matrix fits in memory.

    """
    bm = df.BoundaryMesh(mesh, 'exterior', False)
    N = bm.num_vertices()
    byte_size_float = np.zeros(1, dtype=float).nbytes
    memory_usage = N ** 2 * byte_size_float

    logger.debug("Boundary element matrix for mesh with {} vertices and {} "
                 "surface nodes will occupy {} in memory.".format(
                     mesh.num_vertices(), N, make_human_readable(memory_usage)))


def build_maps(functionspace, dim=3, scalar=False):
    v2d_xyz = df.vertex_to_dof_map(functionspace)
    d2v_xyz = df.dof_to_vertex_map(functionspace)
    n1, n2 = len(v2d_xyz), len(d2v_xyz)

    v2d_xxx = ((v2d_xyz.reshape(n1/dim, dim)).transpose()).reshape(-1,)

    d2v_xxx = d2v_xyz.copy()
    for i in xrange(n2):
        j = d2v_xyz[i]
        d2v_xxx[i] = (j%dim)*n1/dim + (j/dim)

    n = n1 - n2

    """
    #in the presence of pbc, n1 > n2, here we try to reduce the length of v2d_xyz.
    if n>0:

        #next, we reduce the length of v2d_xyz to n2
        a = []
        b = set()
        for x in v2d_xyz:
            if x not in b:
                b.add(x)
                a.append(x)
        assert(len(a) == n2)
        v2d_xyz2 = np.array(a)

        #we need both d2v_xyz and v2d_xyz2 to make sure the values in d2v_xyz is less than n2.
        d2v_xyz2 = d2v_xyz.copy()
        for i in range(n2):
            if d2v_xyz[i]>n2:
                j = v2d_xyz[d2v_xyz[i]]
                for k in range(n2):
                    if v2d_xyz2[k] == j:
                        d2v_xyz2[i] = k
                        break

        v2d_xxx2 = ((v2d_xyz2.reshape(n2/dim, dim)).transpose()).reshape(-1,)

        d2v_xxx2 = d2v_xyz2.copy()
        for i in xrange(n2):
            j = d2v_xyz2[i]
            d2v_xxx2[i] = (j%dim)*n2/dim + (j/dim)
    """

    if scalar:
        return v2d_xyz, d2v_xyz


    #we then build new mappings for order xxx rather xyz


    #return v2d_xyz2, v2d_xxx2, d2v_xyz2, d2v_xxx2
    return v2d_xyz, v2d_xxx, d2v_xyz, d2v_xxx
