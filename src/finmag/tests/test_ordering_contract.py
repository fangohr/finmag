"""Dedicated public field-array component-ordering contract (Task 31).

NO MASTER ANCESTOR (genuinely new under DOLFINx): the component-blocked vs.
interleaved vs. double-converted array-ordering distinction this file guards
against is a DOLFINx-port-only failure mode -- master's dolfin-native arrays
never had a second "backend ordering" to confuse with the public
component-blocked layout, so there is no master test file to restore. This
suite is new to pin the ported ordering contract directly.

CRITICAL-class guard. A wrong ordering fix does not raise, it silently
scrambles physics, so this module pins the legacy component-blocked (``xxx``)
ordering on every public field-array surface with *analytic* layouts that are
provably distinct from the interleaved (backend) and double-converted layouts:

* ``interaction.compute_field()`` returns the owned-vertex-coordinate-ordered
  component-blocked array ``[x(v0)..x(vK), y(v0)..y(vK), z(v0)..z(vK)]``;
* ``EffectiveField.H_eff`` / ``compute()`` accumulate in that same ordering;
* ``LLG.solve`` consumes the already-blocked ``H_eff`` with NO re-conversion
  (the double-conversion regression is the failure mode this file exists to
  catch);
* ``Field.get_numpy_array_debug()`` returns the blocked view and round-trips
  with ``set_with_numpy_array_debug``.

Every asymmetric case below is constructed so blocked, interleaved and
double-converted layouts differ element-for-element -- a uniform or
component-symmetric field could not tell them apart.
"""

import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI

from finmag.energies import Exchange, Zeeman
from finmag.field import Field
from finmag.physics.effective_field import EffectiveField
from finmag.physics.llg import LLG


def _spaces(domain):
    S1 = fem.functionspace(domain, ("Lagrange", 1))
    S3 = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    return S1, S3


# Component-asymmetric applied field: the three components live in disjoint
# magnitude bands so that a component-blocked flat array has all-x entries
# first, all-y next, all-z last -- an interleaved array would mix the bands.
def _banded_field(x):
    return np.vstack((1.0 + x[0], 1.0e3 + x[1], 1.0e6 + x[2]))


def _banded_expected_rows(coords):
    return np.column_stack(
        (1.0 + coords[:, 0], 1.0e3 + coords[:, 1], 1.0e6 + coords[:, 2])
    )


def test_compute_field_is_component_blocked_and_owned_vertex_ordered():
    # A 2x2x2 cube has a non-trivial dof<->vertex permutation, so this also
    # guards the owned-vertex coordinate ordering, not merely the transpose.
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    S1, S3 = _spaces(domain)
    DG = fem.functionspace(domain, ("DG", 0))
    m = Field(S3, (0.6, 0.0, 0.8), name="m")
    Ms = Field(DG, 8.0e5, name="Ms")

    zeeman = Zeeman(_banded_field)
    zeeman.setup(m, Ms)

    field = zeeman.compute_field()
    n = field.size // 3
    x_block, y_block, z_block = field[:n], field[n:2 * n], field[2 * n:]

    # Component-blocking: each third is one clean magnitude band. An interleaved
    # return would put a y- and z-entry inside the first third and fail this.
    assert x_block.max() < 10.0
    assert 1.0e3 <= y_block.min() and y_block.max() < 1.0e3 + 10.0
    assert z_block.min() >= 1.0e6

    # Exact analytic layout, owned-vertex coordinate order.
    coords, _ = m.coords_and_values()
    rows = _banded_expected_rows(coords)
    expected_blocked = rows.T.reshape(-1)
    np.testing.assert_allclose(field, expected_blocked, rtol=1e-12, atol=1e-9)

    # Provably distinct from the interleaved layout the port used to return.
    interleaved = rows.reshape(-1)
    assert not np.allclose(field, interleaved)


def test_effective_field_H_eff_is_blocked_and_is_the_blocked_sum():
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    S1, S3 = _spaces(domain)
    DG = fem.functionspace(domain, ("DG", 0))
    m = Field(S3, (0.6, 0.0, 0.8), name="m")
    Ms = Field(DG, 8.0e5, name="Ms")

    eff = EffectiveField(m, Ms, unit_length=1e-9)
    zeeman = Zeeman(_banded_field, name="Zeeman")
    exchange = Exchange(1.3e-11, name="Exchange")
    eff.add(zeeman)
    eff.add(exchange)

    H_eff = eff.compute()
    # Accumulation preserves the shared blocked ordering (no re-conversion).
    np.testing.assert_allclose(
        H_eff, zeeman.compute_field() + exchange.compute_field(),
        rtol=1e-12, atol=1e-9,
    )

    # get_dolfin_function must invert the blocked ordering to rebuild the
    # Function -- a raw reconstruction would scramble the components.
    fn = eff.get_dolfin_function("Zeeman")
    rebuilt = Field(S3)
    rebuilt.f.x.array[:] = fn.x.array
    coords, _ = rebuilt.coords_and_values()
    np.testing.assert_allclose(
        rebuilt.coords_and_values()[1], _banded_expected_rows(coords),
        rtol=1e-12, atol=1e-9,
    )


def test_llg_solve_consumes_blocked_H_eff_without_double_conversion():
    # Macrospin: uniform m, uniform but component-asymmetric applied field. The
    # asymmetry is essential -- a double-conversion of H_eff scrambles the three
    # components across nodes (turning a uniform field non-uniform), which a
    # component-symmetric field could never reveal.
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    S1, S3 = _spaces(domain)
    llg = LLG(S1, S3, unit_length=1e-9)
    llg.Ms = 8.6e5
    llg.set_alpha(0.1)
    llg.set_m((1.0, 0.0, 0.0), normalise=True)

    Hx, Hy, Hz = 1.0e4, -3.0e4, 7.0e4  # all distinct and nonzero
    llg.effective_field.add(Zeeman((Hx, Hy, Hz), name="Zeeman"))

    dmdt = llg.solve(0.0)
    nodal = dmdt.reshape((3, -1))
    n = nodal.shape[1]

    # H_eff itself is component-blocked: reshape((3, -1)) rows are the uniform
    # (Hx, Hy, Hz), i.e. each row is constant across nodes.
    H_eff_rows = llg.effective_field.H_eff.reshape((3, -1))
    np.testing.assert_allclose(
        H_eff_rows, np.array([[Hx], [Hy], [Hz]]) * np.ones((1, n)),
        rtol=1e-10, atol=1e-6,
    )

    # dm/dt must be uniform across nodes; a double-converted H_eff would make it
    # node-dependent.
    np.testing.assert_allclose(
        nodal, np.broadcast_to(nodal[:, :1], nodal.shape),
        rtol=1e-10, atol=1e-12,
    )

    # Exact: solve()'s dm/dt equals the node-local kernel applied to the
    # *intended* uniform (m, H) -- proving the plumbing delivered the correct,
    # un-scrambled H to the kernel. (Uses the port's own kernel, so this pins
    # the ordering plumbing, not the physics.)
    m_uniform = llg.m_numpy.reshape((3, -1))
    H_uniform = np.array([[Hx], [Hy], [Hz]]) * np.ones((1, n))
    expected = llg._dmdt_numpy(m_uniform, H_uniform).reshape(-1)
    np.testing.assert_allclose(dmdt, expected, rtol=1e-10, atol=1e-12)
    assert np.max(np.abs(dmdt)) > 0.0


def test_get_numpy_array_debug_is_blocked_and_round_trips():
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    _, S3 = _spaces(domain)
    field = Field(S3, _banded_field, name="m")

    debug = field.get_numpy_array_debug()
    # Blocked, not the raw backend-order as_array().
    np.testing.assert_array_equal(debug, field.get_ordered_numpy_array_xxx())
    assert not np.allclose(debug, field.as_array())

    coords, _ = field.coords_and_values()
    np.testing.assert_allclose(
        debug, _banded_expected_rows(coords).T.reshape(-1),
        rtol=1e-12, atol=1e-9,
    )

    # get/set debug are an exact inverse pair (legacy blocked semantics).
    other = Field(S3, (0.0, 0.0, 0.0))
    other.set_with_numpy_array_debug(debug)
    np.testing.assert_allclose(
        other.as_array(), field.as_array(), rtol=1e-12, atol=1e-9
    )
