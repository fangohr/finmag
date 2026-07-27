"""DOLFINx port of the dolfin-free subset of ``finmag.util.helpers``.

This file now lives at its master path
``src/finmag/util/helpers_test.py`` (formerly
``src/finmag/tests/test_util_helpers_dolfinx.py``), so
``git diff b5015c5a..HEAD -- src/finmag/util/helpers_test.py``
shows the port diff directly.

SCOPE NOTE: the DOLFINx *port* in this file does NOT attempt full parity with
master's version of this same file
(``b5015c5a:src/finmag/util/helpers_test.py``, 29 test functions). The eight
functions it does not port are now carried verbatim at the bottom of this file
under the ``NOT PORTED`` banner (marked ``@pytest.mark.not_ported``), so no
master coverage vanished with the move; they run, and fail, in the non-gating
inventory lane. ``finmag.util.helpers`` imports legacy
``dolfin`` at module scope, so the whole module -- and every test that
exercises a helper built on ``dolfin.Function``/``FunctionSpace``/``Mesh``
objects -- is unimportable/unrunnable under DOLFINx. This file covers the
dolfin-FREE subset only: pure-stdlib/pure-numpy helpers, either re-exported
from :mod:`finmag.util.logging_helpers` (which is itself dolfin-free, see
SR1 P2.1/P4-helpers) or reimplemented locally byte-identical to the current
``finmag.util.helpers`` source (per the ratified minimal-diff workaround --
this module cannot simply ``from finmag.util.helpers import X`` because that
import itself drags in ``dolfin``).

ACCOUNTING for all 29 master ``helpers_test.py`` functions, so a reviewer can
check off every one without re-deriving it:

Transcribed above the banner (16, dolfin-free, real assertions run here):
  - test_logging_handler_str          -> logging_handler_str
  - test_logging_status_str           -> logging_status_str
  - test_components                   -> components
  - test_vectors                      -> vectors
  - test_norms                        -> norm
  - test_fnormalise                   -> fnormalise
  - test_angle                        -> angle (+ norm)
  - test_rows_to_columns              -> rows_to_columns
  - test_cartesian_to_spherical       -> cartesian_to_spherical
  - test_pointing_upwards             -> pointing_upwards (+ cartesian_to_spherical)
  - test_pointing_downwards           -> pointing_downwards (+ cartesian_to_spherical)
  - test_TemporaryDirectory           -> TemporaryDirectory
  - test_run_in_tmpdir                -> run_in_tmpdir
  - test_contextmanager_ignored       -> ignored
  - test_run_cmd_with_timeout         -> run_cmd_with_timeout
  - test_set_color_scheme             -> set_color_scheme (+ ansistrm, dolfin-free)

Transcribed above the banner, skip preserved verbatim (5, already
unconditionally skipped IN MASTER for reasons unrelated to DOLFINx --
transcribed for completeness of the accounting, not because DOLFINx changes
their status):
  - test_get_hg_revision_info  -> get_hg_revision_info   (master: "test for hg")
  - test_binary_tarball_name   -> binary_tarball_name     (master: repo commit removed)
  - test_apply_vertexwise      -> apply_vertexwise        (master: "Broken, but not used anywhere")
  - test_jpg2avi               -> jpg2avi                 (master: skipif("True"))
  - test_pvd2avi               -> pvd2avi                 (master: skipif("True"))

Deferred-because-dolfin-dependent, and therefore CARRIED VERBATIM under the
``NOT PORTED`` banner at the bottom of this file (8; the underlying helper
builds/consumes
legacy ``dolfin`` Mesh/FunctionSpace/Function/SubMesh/MeshFunction objects,
or -- for ``crossprod`` -- reads ``dolfin.parameters.reorder_dofs_serial``
directly; none has a DOLFINx port anywhere in the tree yet, so these are
genuine, not "covered elsewhere"):
  - test_vector_valued_function          -> vector_valued_function
  - test_scalar_valued_dg_function       -> scalar_valued_dg_function
  - test_piecewise_on_subdomains         -> piecewise_on_subdomains
  - test_vector_field_from_dolfin_function -> vector_field_from_dolfin_function
  - test_probe                           -> probe
  - test_crossprod                       -> crossprod
  - test_restriction                     -> restriction (+ scalar_valued_function)
  - test_verify_function_space_type      -> verify_function_space_type

16 + 5 + 8 = 29. No genuine-gaps: every master function is either
transcribed (real or skip-preserved) or carried verbatim under the
``NOT PORTED`` banner.

Below the ``NEW under DOLFINx`` banner: the pre-existing (no master ancestor)
coverage of the ``start_logging_to_file`` / re-export import boundary, kept
as-is. Below the ``NOT PORTED`` banner (last section of the file): the eight
carried master functions.
[Claude Opus 4.8]
"""

import importlib.util
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from threading import Timer

import numpy as np
import pytest

# ``finmag.util.logging_helpers`` is pure stdlib (logging/os only, see SR1
# P2.1/P4-helpers) and dolfin-free, so it is safe to import directly here
# (unlike ``finmag.util.helpers``, which drags in legacy ``dolfin``).
from finmag.util.logging_helpers import logging_handler_str, logging_status_str

SRC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SRC_ROOT.parent

requires_dolfinx = pytest.mark.skipif(
    importlib.util.find_spec("dolfinx") is None,
    reason="The dolfin-free logging-helper checks require the DOLFINx environment.",
)


def _run_isolated(code, cwd=None, check=True):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(SRC_ROOT)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd or REPO_ROOT),
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


# ==========================================================================
# MASTER-DERIVED TESTS (faithful transcription of the dolfin-free subset of
# ``b5015c5a:src/finmag/util/helpers_test.py``)
# ==========================================================================

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

TOLERANCE = 1e-15  # master TOLERANCE = 1e-15, verbatim; measured diffs ~0 below

# ``finmag.util.helpers`` imports legacy ``dolfin`` at module scope and is
# therefore unimportable under DOLFINx. The functions below are reimplemented
# here byte-identical to the current ``finmag.util.helpers`` source (which is
# itself unchanged, dolfin-free logic for each of these) per the ratified
# minimal-diff workaround -- see e.g. ``test_llg.py`` /
# ``sim/sim_test.py`` for the same pattern. [Claude Opus 4.8]

def components(vs):
    return vs.view().reshape((3, -1))


def vectors(vs):
    number_of_nodes = len(vs) // 3
    return vs.view().reshape((number_of_nodes, -1), order="F")


def norm(vs):
    if not type(vs) == np.ndarray:
        vs = np.array(vs)
    if vs.shape == (3, ):
        return np.linalg.norm(vs)
    return np.sqrt(np.add.reduce(vs * vs, axis=1))


def fnormalise(arr, ignore_zero_vectors=False):
    a = arr.astype(np.float64)  # this copies

    a = a.reshape((3, -1))
    a_norm = np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    if ignore_zero_vectors:
        # Leave zero vectors untouched instead of dividing them by zero.
        out = a.copy()
    else:
        # Match the historical NaN result for zero vectors, but without a warning.
        out = np.empty_like(a)
        out.fill(np.nan)
    np.divide(a, a_norm, out=out, where=(a_norm != 0))
    return out.ravel()


def angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (norm(v1) * norm(v2)))


def rows_to_columns(arr):
    return arr.reshape(arr.size, order="F").reshape((3, -1))


def cartesian_to_spherical(vector):
    r = np.linalg.norm(vector)
    unit_vector = np.array(vector) / r
    theta = np.arccos(unit_vector[2])
    phi = np.arctan2(unit_vector[1], unit_vector[0])
    return np.array((r, theta, phi))


# dolfin df2: master spells this ``def pointing_upwards((x, y, z)):`` (Python 2
# tuple-parameter unpacking, a SyntaxError under Python 3); converted to
# explicit unpacking in the body, matching current ``finmag.util.helpers``.
def pointing_upwards(coords):
    x, y, z = coords
    _, theta, _ = cartesian_to_spherical((x, y, z))
    return theta <= (np.pi / 4)


def pointing_downwards(coords):
    x, y, z = coords
    _, theta, _ = cartesian_to_spherical((x, y, z))
    return abs(theta - np.pi) < (np.pi / 4)


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
    try:
        yield
    except exceptions:
        pass


def run_cmd_with_timeout(cmd, timeout_sec):
    proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    timer.start()
    stdout, stderr = proc.communicate()
    timer.cancel()
    # dolfin df2/py2->3: current ``finmag.util.helpers.run_cmd_with_timeout``
    # decodes bytes to str (subprocess.PIPE yields bytes under Python 3; master's
    # Python 2 assertion ``stdout == 'hello\n'`` needs this to still hold).
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8")
    return proc.returncode, stdout, stderr


# ``ansistrm`` (used by ``set_color_scheme``) is pure stdlib (ctypes/logging/os)
# and dolfin-free; imported directly here rather than via ``finmag.util.helpers``.
from finmag.util import ansistrm  # noqa: E402

logger = logging.getLogger("finmag")

supported_color_schemes = ansistrm.level_maps.keys()
supported_color_schemes_str = ", ".join(
    ["'{}'".format(s) for s in supported_color_schemes])


def set_color_scheme(color_scheme):
    if color_scheme not in supported_color_schemes:
        raise ValueError(
            "Color scheme must be one of: {}".format(supported_color_schemes_str))
    for h in logger.handlers:
        if not isinstance(h, ansistrm.ColorizingStreamHandler):
            continue
        h.level_map = ansistrm.level_maps[color_scheme]


@requires_dolfinx
def test_logging_handler_str():
    """
    """
    hdlr = logging.NullHandler()
    hdlr_str = logging_handler_str(hdlr)
    print(hdlr_str)
    # RED-first finding, NOT a dolfin/dolfinx difference: master's regex
    # ("^<logging.NullHandler object at .*>$") targets Python 2's fallback
    # `object.__str__`. Under Python 3 the stdlib's `logging.Handler` defines
    # its own `__repr__`/`__str__` (`<NullHandler (NOTSET)>`), so the format
    # genuinely changed for every Python 3 interpreter, independent of this
    # port. Measured: hdlr_str == "<NullHandler (NOTSET)>". Assertion
    # STRUCTURE (regex match on the handler string) kept verbatim; only the
    # literal pattern is updated to match Python 3's real stdlib output.
    assert(re.match(r"^<NullHandler \(.*\)>$", hdlr_str) != None)


@requires_dolfinx
def test_logging_status_str():
    """
    Test that we can call the function `logging_status_str()` and it returns a
    non-empty string.
    """
    status_str = logging_status_str()
    print(status_str)
    assert(isinstance(status_str, str) and (status_str != ""))


@requires_dolfinx
def test_components():
    x = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    y = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    assert np.array_equal(y, components(x))


@requires_dolfinx
def test_vectors():
    x1 = np.array([1, 1, 2, 2, 3, 3])
    y1 = np.array([[1, 2, 3], [1, 2, 3]])
    assert np.array_equal(y1, vectors(x1))

    x2 = np.array([0, 1, 1, 0, 2, 3, 3, 2, 4, 5, 5, 4])
    y2 = np.array([[0, 2, 4], [1, 3, 5], [1, 3, 5], [0, 2, 4]])
    assert np.array_equal(y2, vectors(x2))


@requires_dolfinx
def test_norms():
    v = [1, 1, 0]
    assert abs(norm(v) - np.sqrt(2)) < TOLERANCE  # measured: 0.0

    v = np.array([[1, 1, 0], [1, -2, 3]])
    assert np.allclose(
        norm(v), np.array([np.sqrt(2), np.sqrt(14)]), rtol=TOLERANCE)


@requires_dolfinx
def test_fnormalise():
    a = np.array([1., 1., 2., 2., 0., 0.])
    norm_ = np.sqrt(1 + 2 ** 2 + 0 ** 2)
    expected = a[:] / norm_
    assert np.allclose(fnormalise(a), expected, rtol=TOLERANCE)

    a = np.array([1., 2., 0, 0., 1., 3.])
    n1 = np.sqrt(1 + 0 + 1)
    n2 = np.sqrt(2 ** 2 + 0 + 3 ** 2)
    expected = a[:] / np.array([n1, n2, n1, n2, n1, n2])
    assert np.allclose(fnormalise(a), expected, rtol=TOLERANCE)

    a = np.array([5 * [1.], 5 * [0], 5 * [0]])
    expected = a.copy().ravel()
    assert np.allclose(fnormalise(a), expected, rtol=TOLERANCE)

    a2 = np.array([5 * [2.], 5 * [0], 5 * [0]])
    assert np.allclose(fnormalise(a2), expected, rtol=TOLERANCE)

    # this is 0   0   3   4
    #        0   2   0   5
    #        1   0   0   0
    #
    # can also write as

    a3 = np.array([[0, 0, 1.], [0, 2, 0], [3, 0, 0], [4, 5, 0]]).transpose()

    c = np.sqrt(4 ** 2 + 5 ** 2)
    expected = np.array([0, 0, 1, 4 / c, 0, 1, 0, 5 / c, 1, 0, 0, 0])
    print("a3=\n", a3)
    print("expected=\n", expected)
    print("fnormalise(a3)=\n", fnormalise(a3))
    assert np.allclose(fnormalise(a3), expected, rtol=TOLERANCE)

    # check that normalisation also works if input vector happens to be an
    # integer array
    # first with floats
    a4 = np.array([0., 1., 1.])
    c = np.sqrt(1 ** 2 + 1 ** 2)  # sqrt(2)
    expected = np.array([0, 1 / c, 1 / c])
    print("a4=\n", a4)
    print("expected=\n", expected)
    print("fnormalise(a4)=\n", fnormalise(a4))
    assert np.allclose(fnormalise(a4), expected, rtol=TOLERANCE)

    # the same test with ints (i.e.
    a5 = np.array([0, 1, 1])
    expected = a5 / np.sqrt(2)
    assert np.allclose(fnormalise(a5), expected, rtol=TOLERANCE)

    # test that zero vectors in the input result in NaN if 'ignore_zero_vectors=False'
    a6 = np.array([2, 0, 0, 0, 0, 0])
    a6_normalised = fnormalise(a6)
    assert a6_normalised.shape == (6,)
    assert np.allclose(a6_normalised[[0, 2, 4]], [1, 0, 0])
    assert np.isnan(a6_normalised[[1, 3, 5]]).all()

    # test that zero vectors in the input result in NaN if 'ignore_zero_vectors=False'
    a7 = np.array([3, 0, 4, 0, 0, 0])
    expected = np.array([0.6, 0, 0.8, 0, 0, 0])
    assert np.allclose(fnormalise(a7, ignore_zero_vectors=True), expected, rtol=TOLERANCE)


@requires_dolfinx
def test_angle():
    assert abs(angle([1, 0, 0], [1, 0, 0])) < TOLERANCE
    assert abs(angle([1, 0, 0], [0, 1, 0]) - np.pi / 2) < TOLERANCE
    assert abs(angle([1, 0, 0], [1, 1, 0]) - np.pi / 4) < TOLERANCE


@requires_dolfinx
def test_rows_to_columns():
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    assert np.array_equal(y, rows_to_columns(x))


@requires_dolfinx
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
        print("Testing vector {}. Got {}. Expected {}.".format(v, v_spherical, expected[i]))
        assert np.max(np.abs(v_spherical - expected[i])) < TOLERANCE


@requires_dolfinx
def test_pointing_upwards():
    assert pointing_upwards((0, 0, 1))
    assert pointing_upwards((0.5, 0.5, 0.8))
    assert pointing_upwards((-0.5, 0.5, 0.8))
    assert not pointing_upwards((0, 0, -1))
    assert not pointing_upwards((-0.5, 0.5, -0.8))
    assert not pointing_upwards((-0.5, 0.5, 0.4))


@requires_dolfinx
def test_pointing_downwards():
    assert pointing_downwards((0, 0, -1))
    assert pointing_downwards((-0.5, -0.5, -0.8))
    assert pointing_downwards((-0.5, 0.5, -0.8))
    assert not pointing_downwards((0, 0, 1))
    assert not pointing_downwards((-0.5, -0.5, 0.8))
    assert not pointing_downwards((-0.5, 0.5, -0.4))


@requires_dolfinx
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


@requires_dolfinx
def test_run_in_tmpdir():
    cwd_bak = os.getcwd()

    # Inside the 'with' block we should be in the
    # newly created temporary directory.
    with run_in_tmpdir() as tmpdir:
        assert os.getcwd() == tmpdir
        assert tmpdir != cwd_bak

    # Outside the 'with' block we should be back in
    # the previous working directory and the temporary
    # directory should be destroyed
    assert os.getcwd() == cwd_bak
    assert not os.path.exists(tmpdir)

    # Using 'keep=True' the temporary directory should
    # not be deleted.
    with run_in_tmpdir(keep=True) as tmpdir2:
        pass
    assert os.path.exists(tmpdir2)

    # Tidy up
    os.rmdir(tmpdir2)


@requires_dolfinx
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


@requires_dolfinx
def test_run_cmd_with_timeout():
    # A successfully run command should have exit code 0
    returncode, stdout, _ = run_cmd_with_timeout('echo hello', timeout_sec=100)
    assert(returncode == 0)
    assert stdout == 'hello\n'

    # A non-existing command should raise OSError
    with pytest.raises(OSError):
        returncode, _, _ = run_cmd_with_timeout('foo', timeout_sec=1)

    # This command should be killed due to the timeout, resulting in a return
    # code of -9.
    returncode, _, _ = run_cmd_with_timeout('sleep 10', timeout_sec=0)
    assert(returncode == -9)


@requires_dolfinx
def test_set_color_scheme():
    """
    Just check that we can call 'set_color_scheme' with allowed values.
    We don't check that the color scheme is actually changed, since it's
    not obvious how to do that.

    """
    set_color_scheme('light_bg')
    set_color_scheme('dark_bg')
    set_color_scheme('none')
    with pytest.raises(ValueError):
        set_color_scheme('foobar')


# --------------------------------------------------------------------------
# Skip-preserved master tests: these were ALREADY unconditionally skipped in
# master, for reasons unrelated to DOLFINx (Mercurial tooling gone, a scrubbed
# commit, "Broken, but not used anywhere", or a bare `skipif("True")`).
# Transcribed verbatim (skip marker included) purely so the 29-function
# accounting above is checkable against real test objects; nothing here
# starts passing under DOLFINx that didn't already skip in master.
# --------------------------------------------------------------------------

@pytest.mark.skipif(True, reason="test for hg")
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
    rev_nr, rev_id, rev_date = get_hg_revision_info(
        finmag_repo, revision=id_string)
    assert(rev_nr == 4)
    assert(rev_id == id_string)
    assert(rev_date == '2012-02-02')


@pytest.mark.skip(reason='Commit does not exist anymore (sensitive data removed)')
def test_binary_tarball_name(tmpdir):
    finmag_repo = MODULE_DIR
    expected_tarball_name = 'FinMag-dist__2017-06-16__3127873bac77fbade1ec4ed0f9017b3cb0204a1f_foobar.tar.bz2'
    assert(binary_tarball_name(finmag_repo, revision='3127873bac77fbade1ec4ed0f9017b3cb0204a1f',
                               suffix='_foobar') == expected_tarball_name)


@pytest.mark.skip(reason='Broken, but not used anywhere')
def test_apply_vertexwise():
    xmin = ymin = zmin = -2
    xmax = ymax = zmax = 3
    nx = ny = nz = 10
    mesh = df.BoxMesh(df.Point(xmin, ymin, zmin), df.Point(xmax, ymax, zmax), nx, ny, nz)
    V = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
    u = df.interpolate(df.Expression(['x[0]', 'x[1]', '0'], degree=1), V)
    v = df.interpolate(df.Expression(['-x[1]', 'x[0]', 'x[2]'], degree=1), V)
    w = df.interpolate(
        df.Expression(['x[1]*x[2]', '-x[0]*x[2]', 'x[0]*x[0]+x[1]*x[1]'], degree=1), V)

    uxv = apply_vertexwise(np.cross, u, v)

    assert(np.allclose(uxv.vector().array(), w.vector().array()))


@pytest.mark.skipif("True")
def test_jpg2avi(tmpdir):
    """
    Test whether we can create an animation from a series of .jpg images.

    """
    os.chdir(str(tmpdir))
    sim = finmag.example.normal_modes.disk()
    sim.compute_normal_modes(n_values=3)
    sim.export_normal_mode_animation(k=0, filename='foo/bar.pvd')
    # note that we must not trim the border because otherwise the resulting
    # .jpg files will have different sizes, which confused mencoder
    render_paraview_scene(
        'foo/bar.pvd', outfile='foo/quux.jpg', trim_border=False)

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
    pvd2avi('foo/bar.pvd', outfilename='animation.avi', duration=10,
            fps=10, add_glyphs=False, colormap='heated_body')
    assert(os.path.exists('animation.avi'))


# ===== NEW under DOLFINx (no master ancestor) =====
#
# The tests below have no master ancestor in ``helpers_test.py``: master never
# tested ``start_logging_to_file`` (or the legacy re-export spelling) at all.
# They exercise the SR1 P2.1/P4-helpers extraction of the dolfin-free logging
# helpers into :mod:`finmag.util.logging_helpers` -- ``start_logging_to_file``
# (per-simulation / per-run logfile wiring) plus its two logging-introspection
# companions ``logging_status_str`` and ``logging_handler_str`` -- so they
# resolve without legacy dolfin, while ``finmag.util.helpers`` keeps
# re-exporting them for the legacy spelling.
#
# The ``shutdown`` / ``instances_*`` / ``close_logfile`` surface named by the
# scoping investigation is deliberately NOT covered here: those are
# ``Simulation`` *methods* in the legacy ``sim.py`` (coupled to the tablewriter /
# scheduler cyclic-reference teardown and ``Simulation.instances`` bookkeeping),
# not ``finmag.util.helpers`` functions, and no standalone
# ``instances`` / ``get_instances`` / ``finmag_instances`` helper exists anywhere
# in the legacy tree. There is no ``finmag.util.helpers`` symbol to extract for
# them, so they stay out of this helpers-extraction slice.
#
# Everything runs in a clean subprocess so ``sys.modules`` is an exact witness of
# what was actually imported. [Claude Opus 4.8]


# --------------------------------------------------------------------------
# importability, dolfin-free
# --------------------------------------------------------------------------

@requires_dolfinx
def test_logging_helpers_import_without_legacy_dolfin():
    """The extracted helpers resolve from the stdlib-only module with no
    legacy ``dolfin`` anywhere in ``sys.modules``."""
    _run_isolated(
        """
import sys
from finmag.util.logging_helpers import (
    start_logging_to_file, logging_status_str, logging_handler_str,
)

assert start_logging_to_file.__module__ == "finmag.util.logging_helpers"
assert logging_status_str.__module__ == "finmag.util.logging_helpers"
assert logging_handler_str.__module__ == "finmag.util.logging_helpers"

assert "dolfin" not in sys.modules
assert "dolfinx" not in sys.modules
"""
    )


# --------------------------------------------------------------------------
# start_logging_to_file: real behaviour (attach handler + write records)
# --------------------------------------------------------------------------

@requires_dolfinx
def test_start_logging_to_file_attaches_handler_and_writes_records(tmp_path):
    """``start_logging_to_file`` adds a file handler to the ``finmag`` logger,
    returns it, and finmag log records land in the given file -- all without
    legacy dolfin."""
    result = _run_isolated(
        """
import logging
import sys
from finmag.util.logging_helpers import start_logging_to_file, logging_handler_str

logger = logging.getLogger("finmag")
logger.setLevel(logging.DEBUG)

before = list(logger.handlers)
handler = start_logging_to_file("run.log", mode="w", level=logging.DEBUG)

# The returned object is a logging handler now attached to the finmag logger.
assert isinstance(handler, logging.Handler)
assert handler in logger.handlers
assert handler not in before

# logging_handler_str names the backing file for a file handler.
assert logging_handler_str(handler).endswith("run.log")

logger.info("HELLO_MARKER_P4HELPERS")
handler.flush()

assert "dolfin" not in sys.modules
print("OK")
""",
        cwd=tmp_path,
    )
    assert "OK" in result.stdout
    logfile = tmp_path / "run.log"
    assert logfile.exists()
    assert "HELLO_MARKER_P4HELPERS" in logfile.read_text()


@requires_dolfinx
def test_start_logging_to_file_creates_missing_directories(tmp_path):
    """Legacy contract: the helper creates any missing parent directories for
    the logfile before opening it."""
    result = _run_isolated(
        """
import logging
import os
import sys
from finmag.util.logging_helpers import start_logging_to_file

logger = logging.getLogger("finmag")
logger.setLevel(logging.DEBUG)

path = os.path.join("nested", "deeper", "run.log")
assert not os.path.exists("nested")
handler = start_logging_to_file(path, mode="w")
logger.info("NESTED_MARKER")
handler.flush()
assert os.path.exists(path)
assert "dolfin" not in sys.modules
print("OK")
""",
        cwd=tmp_path,
    )
    assert "OK" in result.stdout
    nested = tmp_path / "nested" / "deeper" / "run.log"
    assert nested.exists()
    assert "NESTED_MARKER" in nested.read_text()


@requires_dolfinx
def test_closing_the_logfile_handler_stops_records(tmp_path):
    """The handler returned by ``start_logging_to_file`` can be torn down the
    way the legacy ``Simulation.close_logfile`` did -- close its stream and
    remove it from the ``finmag`` logger -- after which new records no longer
    reach the file. This exercises the teardown contract that the (deferred,
    Simulation-bound) ``shutdown`` path relied on, using only the extracted
    helper's return value."""
    result = _run_isolated(
        """
import logging
import sys
from finmag.util.logging_helpers import start_logging_to_file

logger = logging.getLogger("finmag")
logger.setLevel(logging.DEBUG)

handler = start_logging_to_file("run.log", mode="w")
logger.info("BEFORE_CLOSE")
handler.flush()

# Legacy close_logfile behaviour: close the stream, drop the handler.
handler.stream.close()
logger.removeHandler(handler)
assert handler not in logger.handlers

logger.info("AFTER_CLOSE")
assert "dolfin" not in sys.modules
print("OK")
""",
        cwd=tmp_path,
    )
    assert "OK" in result.stdout
    contents = (tmp_path / "run.log").read_text()
    assert "BEFORE_CLOSE" in contents
    assert "AFTER_CLOSE" not in contents


# --------------------------------------------------------------------------
# logging_status_str / logging_handler_str: real behaviour
# --------------------------------------------------------------------------

@requires_dolfinx
def test_logging_status_and_handler_str_report_the_finmag_logger(tmp_path):
    _run_isolated(
        """
import logging
import sys
from finmag.util.logging_helpers import (
    start_logging_to_file, logging_status_str, logging_handler_str,
)

# A plain stream handler stringifies to its stream.
stream_handler = logging.StreamHandler()
assert logging_handler_str(stream_handler) == str(stream_handler.stream)

logger = logging.getLogger("finmag")
logger.setLevel(logging.DEBUG)
file_handler = start_logging_to_file("status.log", mode="w")

status = logging_status_str()
assert isinstance(status, str)
assert "Current logging status" in status
# The freshly attached file handler shows up by its backing filename.
assert "status.log" in status

assert "dolfin" not in sys.modules
""",
        cwd=tmp_path,
    )


# --------------------------------------------------------------------------
# legacy re-export spelling
# --------------------------------------------------------------------------

def test_helpers_still_reexports_logging_file_helpers():
    """The legacy lane imports these helpers from ``finmag.util.helpers``; the
    move to the stdlib-only module keeps that spelling working and pointing at
    the same objects. ``finmag.util.helpers`` itself still needs legacy dolfin,
    so this only runs in the legacy environment (mirroring the P2.1
    ``set_logging_level`` re-export guard)."""
    if importlib.util.find_spec("dolfin") is None:
        pytest.skip("finmag.util.helpers itself still needs legacy dolfin.")
    from finmag.util import helpers, logging_helpers

    assert helpers.start_logging_to_file is logging_helpers.start_logging_to_file
    assert helpers.logging_status_str is logging_helpers.logging_status_str
    assert helpers.logging_handler_str is logging_helpers.logging_handler_str


# ===== NOT PORTED (carried verbatim from master b5015c5a; expected to fail) =====
# The eight master ``helpers_test.py`` functions the accounting above lists as
# "deferred-because-dolfin-dependent": each drives a helper that builds or
# consumes legacy ``dolfin`` Mesh/FunctionSpace/Function/SubMesh/MeshFunction
# objects (or, for ``crossprod``, reads ``dolfin.parameters.reorder_dofs_serial``
# directly), and none has a DOLFINx port anywhere in the tree. They are carried
# here verbatim from ``git show b5015c5a:src/finmag/util/helpers_test.py`` and
# marked ``@pytest.mark.not_ported`` plus a strict
# ``@pytest.mark.xfail(reason="not ported: ... (deferred: finmag.util.helpers
# dolfin-free port)", strict=True)`` (SR1 S0, owner decision 2026-07-27) so the
# focused ``dolfinx-src-import-pytest`` gate runs them unfiltered and reports
# them as xfailed, while the non-gating inventory lane runs -- and reports --
# them as failures. [owner 2026-07-27]
#
# Master's module-level ``import dolfin as df`` and
# ``from finmag.util.helpers import *`` cannot run under DOLFINx (both raise at
# import time, which would break collection of this whole file), so they are
# guarded below. Only the names these eight functions actually need are bound,
# and only names this module does NOT already define itself -- in particular
# ``fnormalise`` stays the local, byte-identical reimplementation above rather
# than being clobbered by a star import.
try:  # guarded: master had a bare ``import dolfin as df`` at module scope
    import dolfin as df
except ImportError:  # pragma: no cover - the DOLFINx environment has no dolfin
    df = None  # the not_ported tests below fail visibly at call time

try:  # guarded: master had ``from finmag.util.helpers import *``
    from finmag.util.helpers import (  # noqa: F401
        crossprod,
        piecewise_on_subdomains,
        probe,
        restriction,
        scalar_valued_dg_function,
        scalar_valued_function,
        vector_field_from_dolfin_function,
        vector_valued_function,
        verify_function_space_type,
    )
except ImportError:  # pragma: no cover - finmag.util.helpers imports dolfin
    crossprod = piecewise_on_subdomains = probe = restriction = None
    scalar_valued_dg_function = scalar_valued_function = None
    vector_field_from_dolfin_function = vector_valued_function = None
    verify_function_space_type = None

# Unguarded (master had these module-level too and both import cleanly under
# DOLFINx; they simply return dolfinx meshes, so the carried tests fail at call
# time rather than at import time).
from finmag.util.meshes import box, cylinder  # noqa: E402,F401
from finmag.util.mesh_templates import Sphere  # noqa: E402,F401


@pytest.mark.xfail(reason="not ported: finmag.util.helpers.vector_valued_function (deferred: finmag.util.helpers dolfin-free port)", strict=True)
@pytest.mark.not_ported
def test_vector_valued_function():
    """
    Test that the different ways of initialising a vector-valued
    function on a 3d mesh work and that they produce the expected
    results.

    """
    mesh = df.UnitCubeMesh(2, 2, 2)
    # shift mesh coords to avoid dividing by zero when normalising below
    mesh.coordinates()[:] = mesh.coordinates() + 1.0
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
    v_ref_expr = (mesh.coordinates() * [a, b, c]).transpose().reshape((-1,))
    v_ref_expr_normalised = fnormalise(v_ref_expr)

    # Create functions using the various methods
    f_tuple = vector_valued_function(tuple(vec), S3)  # 3-tuple
    f_list = vector_valued_function(list(vec), S3)  # 3-list
    # numpy array representing a 3-vector
    f_array3 = vector_valued_function(np.array(vec), S3)
    # df. Constant representing a 3-vector
    f_dfconstant = vector_valued_function(df.Constant(vec), S3)
    # tuple of strings (will be cast to df.Expression)
    f_expr = vector_valued_function(
        ('a*x[0]', 'b*x[1]', 'c*x[2]'), S3, a=a, b=b, c=c)
    # numpy array of nodal values shape (3*n,)
    f_array3xN = vector_valued_function(v_ref, S3)
    f_arrayN3 = vector_valued_function(
        np.array([vec for r in mesh.coordinates()]), S3)  # numpy array of shape (n, 3)
    # f_callable = vector_valued_function(lambda coords: v_ref_expr, S3) # callable accepting mesh node coordinates and yielding the function values
    # Cython 0.17.1 does not like this
    # f_callable = vector_valued_function(lambda (x,y,z): (a*x, b*y, c*z), S3) # callable accepting mesh node coordinates and yielding the function values
    # but this one is okay
    # callable accepting mesh node coordinates and yielding the function values
    f_callable = vector_valued_function(
        lambda t: (a * t[0], b * t[1], c * t[2]), S3)

    # A few normalised versions, too
    f_tuple_normalised = vector_valued_function(tuple(vec), S3, normalise=True)
    f_expr_normalised = vector_valued_function(
        ('a*x[0]', 'b*x[1]', 'c*x[2]'), S3, a=a, b=b, c=c, normalise=True)

    # Cython 0.17.1 does not like this
    #f_callable_normalised = vector_valued_function(lambda (x,y,z): (a*x, b*y, c*z), S3, normalise=True)
    # but accepts this rephrased version:
    f_callable_normalised = vector_valued_function(
        lambda t: (a * t[0], b * t[1], c * t[2]), S3, normalise=True)

    # Check that the function vectors are as expected
    #import ipdb; ipdb.set_trace()
    assert(all(f_tuple.vector() == v_ref))
    assert(all(f_list.vector() == v_ref))
    assert(all(f_array3.vector() == v_ref))
    assert(all(f_dfconstant.vector() == v_ref))
    assert(all(f_expr.vector() == v_ref_expr))
    assert(all(f_array3xN.vector() == v_ref))
    assert(all(f_arrayN3.vector() == v_ref))
    assert(all(f_callable.vector() == v_ref_expr))

    assert(all(f_tuple_normalised.vector() == v_ref_normalised))
    # py2 print statement -> py3 print() call (2to3-level; SyntaxError
    # otherwise, which would break collection of this whole file).
    print("[DDD] #1: {}".format(f_expr_normalised.vector().array()))
    print("[DDD] #2: {}".format(v_ref_expr_normalised))

    assert(all(f_expr_normalised.vector() == v_ref_expr_normalised))
    assert(all(f_callable_normalised.vector() == v_ref_expr_normalised))


@pytest.mark.xfail(reason="not ported: finmag.util.helpers.scalar_valued_dg_function (deferred: finmag.util.helpers dolfin-free port)", strict=True)
@pytest.mark.not_ported
def test_scalar_valued_dg_function():
    mesh = df.UnitCubeMesh(2, 2, 2)

    def init_f(coord):
        x, y, z = coord
        if z <= 0.5:
            return 1
        else:
            return 10

    f = scalar_valued_dg_function(init_f, mesh)

    assert f(0, 0, 0.51) == 10.0
    assert f(0.5, 0.7, 0.51) == 10.0
    assert f(0.4, 0.3, 0.96) == 10.0
    assert f(0, 0, 0.49) == 1.0
    fa = f.vector().array().reshape(2, -1)

    assert np.min(fa[0]) == np.max(fa[0]) == 1
    assert np.min(fa[1]) == np.max(fa[1]) == 10

    dg = df.FunctionSpace(mesh, "DG", 0)
    dgf = df.Function(dg)
    dgf.vector()[0] = 9.9
    f = scalar_valued_dg_function(dgf, mesh)
    assert f.vector().array()[0] == 9.9


@pytest.mark.xfail(reason="not ported: finmag.util.helpers.piecewise_on_subdomains (deferred: finmag.util.helpers dolfin-free port)", strict=True)
@pytest.mark.not_ported
def test_piecewise_on_subdomains():
    """
    Define a simple cubic mesh with three subdomains, create a function
    which takes different values on these subdomains and check that the
    resulting function really has the right values.
    """
    mesh = df.UnitCubeMesh(1, 1, 1)
    fun_vals = (42, 23, -3.14)
    g = df.MeshFunction('size_t', mesh, 3)
    g.array()[:] = [1, 1, 2, 3, 1, 3]
    p = piecewise_on_subdomains(mesh, g, fun_vals)
    # check that p is a proper Function, not a MeshFunction
    assert(isinstance(p, df.Function))
    assert(
        np.allclose(p.vector().array(), np.array([42, 42, 23, -3.14, 42, -3.14])))


@pytest.mark.xfail(reason="not ported: finmag.util.helpers.vector_field_from_dolfin_function (deferred: finmag.util.helpers dolfin-free port)", strict=True)
@pytest.mark.not_ported
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
                       '0.3 - 0.8*x[0] - 5*x[1] + 0.2*x[2]'), degree=1)
    f = df.interpolate(e, V)

    X, Y, Z = np.mgrid[xmin:xmax:nx * 1j, ymin:ymax:ny * 1j, zmin:zmax:nz * 1j]

    # Evaluate the vector field on the grid to create the reference arrays.
    U = -1.0 - 3 * X + Y
    V = +1.0 + 4 * Y - Z
    W = 0.3 - 0.8 * X - 5 * Y + 0.2 * Z

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


@pytest.mark.xfail(reason="not ported: finmag.util.helpers.probe (deferred: finmag.util.helpers dolfin-free port)", strict=True)
@pytest.mark.not_ported
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
    f = df.interpolate(df.Expression(['x[0]', '0', '0'], degree=1), V)

    # Define the probing points along the x-axis
    xs = np.linspace(-9.9, 9.9, 20)
    pts = [[x, 0, 0] for x in xs]

    def square_x_coord(pt):
        return pt[0] ** 2

    # Probe the field (once normally and once with an additional
    # function applied to the result). Note that the results have
    # different shapes because apply_func returns a scalar, not a
    # 3-vector.
    res1 = probe(f, pts)
    res2 = probe(f, pts, apply_func=square_x_coord)

    # Check that we get the expected results.
    res1_expected = [[x, 0, 0] for x in xs]
    res2_expected = xs ** 2
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


@pytest.mark.xfail(reason="not ported: finmag.util.helpers.crossprod (deferred: finmag.util.helpers dolfin-free port)", strict=True)
@pytest.mark.not_ported
def test_crossprod():
    """
    Compute the cross product of two functions f and g numerically
    using `helpers.crossprod` and compare with the analytical
    expression.

    """
    xmin = ymin = zmin = -2
    xmax = ymax = zmax = 3
    nx = ny = nz = 10
    mesh = df.BoxMesh(df.Point(xmin, ymin, zmin), df.Point(xmax, ymax, zmax), nx, ny, nz)
    V = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
    u = df.interpolate(df.Expression(['x[0]', 'x[1]', '0'], degree=1), V)
    v = df.interpolate(df.Expression(['-x[1]', 'x[0]', 'x[2]'], degree=1), V)
    w = df.interpolate(
        df.Expression(['x[1]*x[2]', '-x[0]*x[2]', 'x[0]*x[0]+x[1]*x[1]'], degree=1), V)

    a = u.vector().array()
    b = v.vector().array()
    c = w.vector().array()

    axb = crossprod(a, b)
    assert(np.allclose(axb, c))


@pytest.mark.xfail(reason="not ported: finmag.util.helpers.restriction (deferred: finmag.util.helpers dolfin-free port)", strict=True)
@pytest.mark.not_ported
def test_restriction(tmpdir):
    """
    Create a mesh consisting of two separate regions and define a
    dolfin Function on it which is constant in either region.
    Then extract the two subfunctions corresponding to these regions
    and check that they are constant and their function vectors
    have the correct lengths.

    """
    os.chdir(str(tmpdir))
    sphere1 = Sphere(10, center=(-20, 0, 0), name="sphere1")
    sphere2 = Sphere(20, center=(+30, 0, 0), name="sphere2")

    mesh = (sphere1 + sphere2).create_mesh(maxh=5.0)

    class Sphere1(df.SubDomain):

        def inside(self, pt, on_boundary):
            return pt[0] < 0

    class Sphere2(df.SubDomain):

        def inside(self, pt, on_boundary):
            return pt[0] > 0
    region_markers = df.CellFunction('size_t', mesh)
    subdomain1 = Sphere1()
    subdomain2 = Sphere2()
    subdomain1.mark(region_markers, 1)
    subdomain2.mark(region_markers, 2)

    submesh1 = df.SubMesh(mesh, region_markers, 1)
    submesh2 = df.SubMesh(mesh, region_markers, 2)

    r1 = restriction(mesh, submesh1)
    r2 = restriction(mesh, submesh2)

    # Define a Python function which is constant in either subregion
    def fun_f(pt):
        return 42.0 if (pt[0] < 0) else 23.0

    # Convert the Python function to a dolfin.Function
    f = scalar_valued_function(fun_f, mesh)

    # Restrict the function to each of the subregions
    f1 = r1(f)
    f2 = r2(f)

    assert(np.allclose(f1.vector().array(), 42.0))
    assert(np.allclose(f2.vector().array(), 23.0))
    assert(len(f1.vector().array()) == submesh1.num_vertices())
    assert(len(f2.vector().array()) == submesh2.num_vertices())

    a = f.vector().array()
    a1 = r1(a)
    a2 = r2(a)
    assert(set(a) == set([23.0, 42.0]))
    assert(np.allclose(a1, 42.0))
    assert(np.allclose(a2, 23.0))
    assert(len(a1) == submesh1.num_vertices())
    assert(len(a2) == submesh2.num_vertices())

    # Check a multi-dimensional array, too
    b = np.concatenate([a, a])
    b.shape = (2, -1)
    b1 = r1(b)
    b2 = r2(b)
    assert(set(b.ravel()) == set([23.0, 42.0]))
    assert(np.allclose(b1, 42.0))
    assert(np.allclose(b2, 23.0))
    assert(b1.shape == (2, submesh1.num_vertices()))
    assert(b2.shape == (2, submesh2.num_vertices()))


@pytest.mark.xfail(reason="not ported: finmag.util.helpers.verify_function_space_type (deferred: finmag.util.helpers dolfin-free port)", strict=True)
@pytest.mark.not_ported
def test_verify_function_space_type():
    N = 10
    mesh1d = df.UnitIntervalMesh(N)
    mesh2d = df.UnitSquareMesh(N, N)
    mesh3d = df.UnitCubeMesh(N, N, N)

    V1 = df.FunctionSpace(mesh3d, 'DG', 0)
    V2 = df.VectorFunctionSpace(mesh1d, 'DG', 0, dim=1)
    V3 = df.VectorFunctionSpace(mesh2d, 'CG', 1, dim=3)

    # Check that verifying the known function space types works as expected
    assert(verify_function_space_type(V1, 'DG', 0, dim=None))
    assert(verify_function_space_type(V2, 'DG', 0, dim=1))
    assert(verify_function_space_type(V3, 'CG', 1, dim=3))

    # Check that the verification function returns 'False' if we pass in a
    # non-matching function space type.
    # wrong 'dim' (should be None)
    assert(not verify_function_space_type(V1, 'DG', 0, dim=1))
    # wrong degree
    assert(not verify_function_space_type(V1, 'DG', 1, dim=None))
    # wrong family
    assert(not verify_function_space_type(V1, 'CG', 0, dim=None))

    # wrong 'dim' (should be 1)
    assert(not verify_function_space_type(V2, 'DG', 0, dim=None))
    # wrong 'dim' (should be 1)
    assert(not verify_function_space_type(V2, 'DG', 0, dim=42))
    assert(not verify_function_space_type(V2, 'DG', 42, dim=1))  # wrong degree
    assert(not verify_function_space_type(V2, 'CG', 0, dim=1))  # wrong family

    # wrong dimension
    assert(not verify_function_space_type(V3, 'CG', 1, dim=42))
    assert(not verify_function_space_type(V3, 'CG', 42, dim=1))  # wrong degree
    assert(not verify_function_space_type(V3, 'DG', 1, dim=3))  # wrong family
