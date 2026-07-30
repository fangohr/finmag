"""Nmag EXCHANGE-field comparison under DOLFINx (minimal-diff transcription).

This file has two clearly separated parts (see
``src/finmag/energies/demag/fk_demag_test.py``, formerly ``test_fk_demag_dolfinx.py``,
for the established convention):

1. A MINIMAL-DIFF transcription of master's ``test_against_nmag`` from
   ``src/finmag/tests/comparison/exchange/test_exchange_field.py`` (git
   ``b5015c5a``). Function names, order, and assertion structure are kept
   identical to master; the only differences are (a) dolfin->dolfinx API
   changes (annotated inline), (b) the py2->py3 ``print`` conversion, and
   (c) explanatory comments -- see below for the two unavoidable exceptions.
   Master's tolerance (``REL_TOLERANCE = 2e-14``) passes VERBATIM under
   DOLFINx (measured max ``rel_diff`` ~1.41e-14, recorded inline).

   Master's file also contains ``test_against_oommf`` (an OOMMF comparison).
   That function was OUT OF SCOPE for this port's original task assignment
   (the Nmag comparison only), but is now carried verbatim below, under the
   ``NOT PORTED`` banner (marked ``@pytest.mark.not_ported``): the
   canonical-test-paths move onto master's path requires every master
   function to be accounted for, and ``finmag.util.oommf`` is not yet ported
   (register N62/C20), so it fails visibly there in the non-gating inventory
   lane instead of silently disappearing.

   Two exceptions to "only (a)/(b)/(c) diffs", both necessary just to make
   the file importable/runnable, not behavioural:
     * ``finmag.util.helpers`` (source of master's ``vectors``/``norm``/
       ``stats``/``sphinx_sci``) does ``import dolfin as df`` at module scope
       and is therefore not importable in this DOLFINx env. ``vectors``/
       ``norm`` are reimplemented locally below, logic byte-identical to the
       originals.
     * Master's fixture used ``request.cached_setup(setup=..., teardown=...,
       scope="module")``, a pytest API removed from modern pytest. Replaced
       with the equivalent ``@pytest.fixture(scope="module")``. The dropped
       ``teardown_finmag``/``start_table``/``table_delim``/``table_entries``
       machinery only built a Sphinx-docs RST table (``table.rst``, never
       checked into git -- confirmed via ``git ls-files``) with no assertion
       content, so it is not ported; the corresponding table-append line in
       ``test_against_nmag`` is likewise dropped (commented where it was).

2. A NEW-under-DOLFINx invariant guard, below the ``NEW under DOLFINx``
   banner, with no master ancestor.

[Claude Opus 4.8], [Claude Sonnet 5]
"""

import os

import numpy as np
import pytest

import dolfinx.mesh as dm
import dolfinx.fem as fem
from mpi4py import MPI

from finmag.field import Field
from finmag.energies import Exchange

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

x0 = 0
x1 = 20e-9
xn = 10
Ms = 0.86e6
A = 1.3e-11


def _vectors(vs):
    """dolfin-free port of ``finmag.util.helpers.vectors`` (see module note)."""
    number_of_nodes = len(vs) // 3
    return vs.view().reshape((number_of_nodes, -1), order="F")


def m_gen(r):
    x = np.maximum(np.minimum(r[0] / x1, 1.0), 0.0)
    mx = (2 * x - 1) * 2 / 3
    mz = np.sin(2 * np.pi * x) / 2
    my = np.sqrt(1.0 - mx ** 2 - mz ** 2)
    return np.array([mx, my, mz])


def setup_finmag():
    # df.IntervalMesh(xn, x0, x1) -> dolfinx.mesh.create_interval(comm, xn, [x0, x1])
    mesh = dm.create_interval(MPI.COMM_WORLD, xn, [x0, x1])
    # df mesh.coordinates() returned shape (n, 1) for an IntervalMesh, and
    # master's `zip(*mesh.coordinates())` transposed that to one row of x
    # values, shape (1, n). DOLFINx `mesh.geometry.x` is always (n, 3)
    # (y/z columns are zero for a 1D mesh); take column 0 and row-wrap it to
    # reproduce master's (1, n) shape for `m_gen`.
    coords = np.array([mesh.geometry.x[:, 0]])

    # df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3) -> functionspace with shape=(3,)
    S3 = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    m = Field(S3)
    m.set_with_numpy_array_debug(m_gen(coords).flatten())

    exchange = Exchange(A)
    # df.FunctionSpace(mesh, 'DG', 0) -> functionspace(mesh, ("DG", 0))
    exchange.setup(m, Field(fem.functionspace(mesh, ("DG", 0)), Ms))

    # df.Function(S3); H_exc.vector()[:] = exchange.compute_field() -> DOLFINx
    # Exchange.compute_field() already returns the flat numpy array directly,
    # so there is no dolfin Function/vector wrapper step to reproduce.
    H_exc = exchange.compute_field()
    return dict(m=m, H=H_exc)


@pytest.fixture(scope="module")
def finmag():
    # Replaces master's `request.cached_setup(setup=setup_finmag,
    # teardown=teardown_finmag, scope="module")` (removed pytest API, see
    # module note). No teardown is needed: the only teardown action
    # (writing table.rst) is dropped along with it, see module note.
    return setup_finmag()


def test_against_nmag(finmag):
    REL_TOLERANCE = 2e-14  # master's tolerance, kept verbatim -- passes (see measured value below)

    m_ref = np.genfromtxt(os.path.join(MODULE_DIR, "m0_nmag.txt"))
    m_computed = _vectors(finmag["m"].get_numpy_array_debug())
    assert m_ref.shape == m_computed.shape

    H_ref = np.genfromtxt(os.path.join(MODULE_DIR, "H_exc_nmag.txt"))
    H_computed = _vectors(finmag["H"])  # master: finmag["H"].vector().array()
    assert H_ref.shape == H_computed.shape

    assert m_ref.shape == H_ref.shape
    m_cross_H_ref = np.cross(m_ref, H_ref)
    m_cross_H_computed = np.cross(m_computed, H_computed)

    diff = np.abs(m_cross_H_ref - m_cross_H_computed)
    # master: max([norm(v) for v in m_cross_H_ref]); helpers.norm(v) on a
    # single (3,) vector reduces to np.linalg.norm(v) (see module note on why
    # helpers.norm itself is not imported here).
    rel_diff = diff / max(np.linalg.norm(v) for v in m_cross_H_ref)

    # master appended a row to the (dropped, see module note) Sphinx table here.

    print("comparison with nmag, m x H, relative difference:")
    print("    min, median, max = {}, {}, {}\n    mean, std = {}, {}".format(
        np.min(rel_diff), np.median(rel_diff), np.max(rel_diff),
        np.mean(rel_diff), np.std(rel_diff)))
    # measured DOLFINx max(rel_diff): 1.41e-14 -- passes master's 2e-14 verbatim.
    assert np.max(rel_diff) < REL_TOLERANCE


if __name__ == '__main__':
    f = setup_finmag()
    test_against_nmag(f)


# ==========================================================================
# ===== NEW under DOLFINx (no master ancestor) ============================
# ==========================================================================

def test_interval_mesh_vertex_order_is_ascending():
    """Guards the row-for-row pairing in ``test_against_nmag`` above.

    That comparison pairs finmag's computed ``m``/``H`` against the Nmag
    reference files row-for-row, relying on the DOLFINx interval mesh
    emitting its ``xn + 1`` vertices in deterministic ascending-x order (which
    matches the Nmag reference row order). This is legitimate on this
    structured 1D mesh -- unlike the unstructured 3D Magpar case, where node
    order is NOT preserved under DOLFINx and the port there had to switch to a
    coordinate-based match instead (see
    ``test_exchange_compare_magpar.py``, formerly
    ``test_exchange_compare_magpar_dolfinx.py``). This test pins the
    assumption so a future mesh-generator change that reorders vertices fails
    loudly here instead of silently corrupting the comparison above.
    """
    mesh = dm.create_interval(MPI.COMM_WORLD, xn, [x0, x1])
    coords_x = mesh.geometry.x[:, 0]
    assert np.all(np.diff(coords_x) > 0), "interval vertices not ascending"


# ==========================================================================
# ===== NOT PORTED (carried verbatim from master b5015c5a; expected to fail) =====
# finmag.util.oommf is not ported (manifest N62, capability C20). Runs — and
# fails — in the inventory lane so the gap stays visible. [owner 2026-07-27]
# ==========================================================================
#
# Transcribed verbatim from master's ``test_against_oommf``
# (src/finmag/tests/comparison/exchange/test_exchange_field.py, git b5015c5a),
# including its function-local ``from finmag.util.oommf import ...`` /
# ``from finmag.util.oommf.comparison import ...`` imports. Those are NOT
# additionally guarded at module level: in master they are already local to
# this function (not module-level), so they cannot break collection of this
# file -- they only run, and fail, when this test is actually called.
# Confirmed failure mode in this environment: ``ModuleNotFoundError: No
# module named 'dolfin'`` (finmag.util.oommf -> oommf_calculator ->
# finmag.util.helpers -> ``import dolfin``).
# Mechanical python2->python3 fixes only (2to3-level, no behaviour change):
# ``print "..."`` / ``print stats(rel_diff)`` statements -> ``print(...)``
# calls. ``finmag["table"]``/``table_entries``/``s`` (sphinx_sci)/``stats``
# referenced below are master's dropped Sphinx-table machinery and
# ``finmag.util.helpers`` imports (see module docstring) and are not defined
# in this file; they are unreachable in practice because the oommf import
# above fails first, but are left exactly as master wrote them (verbatim
# transcription).
@pytest.mark.xfail(reason="not ported: finmag.util.oommf comparison (manifest N62/C20)", strict=True)
@pytest.mark.not_ported
def test_against_oommf(finmag):
    REL_TOLERANCE = 8e-2

    from finmag.util.oommf import mesh, oommf_uniform_exchange
    from finmag.util.oommf.comparison import oommf_m0, finmag_to_oommf

    oommf_mesh = mesh.Mesh((xn, 1, 1), size=(x1, 1e-12, 1e-12))
    oommf_exc = oommf_uniform_exchange(oommf_m0(m_gen, oommf_mesh), Ms, A).flat
    finmag_exc = finmag_to_oommf(finmag["H"], oommf_mesh, dims=1)

    assert oommf_exc.shape == finmag_exc.shape
    diff = np.abs(oommf_exc - finmag_exc)
    rel_diff = diff / \
        np.sqrt(
            np.max(oommf_exc[0] ** 2 + oommf_exc[1] ** 2 + oommf_exc[2] ** 2))

    finmag["table"] += table_entries.format(
        "oommf", s(REL_TOLERANCE, 0), s(np.max(rel_diff)), s(np.mean(rel_diff)), s(np.std(rel_diff)))

    print("comparison with oommf, H, relative_difference:")  # py2 print stmt -> py3 print()
    print(stats(rel_diff))  # py2 print stmt -> py3 print()
    assert np.max(rel_diff) < REL_TOLERANCE
