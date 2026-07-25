"""DOLFINx port of the dolfin-free subset of ``finmag.util.helpers``.

SCOPE NOTE (the filename overclaims): despite the name, this file does NOT
attempt full parity with master's ``src/finmag/util/helpers_test.py``
(``b5015c5a``, 29 test functions). ``finmag.util.helpers`` imports legacy
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

Deferred-because-dolfin-dependent (8; the underlying helper builds/consumes
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
transcribed (real or skip-preserved) or named above as dolfin-dependent.

Below the banner: the pre-existing (no master ancestor) coverage of the
``start_logging_to_file`` / re-export import boundary, kept as-is.
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
# minimal-diff workaround -- see e.g. ``test_llg_dolfinx.py`` /
# ``test_simulation_dolfinx.py`` for the same pattern. [Claude Opus 4.8]

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
