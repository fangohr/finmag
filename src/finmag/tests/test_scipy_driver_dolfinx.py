"""Focused production tests for the direct DOLFINx SciPy driver (Task 8).

Ports the trusted invariants of the legacy oracle tests
``src/finmag/drivers/tests/test_scipy.py``,
``src/finmag/drivers/tests/test_integrators.py``, and
``src/finmag/drivers/tests/sundials_reinit_test.py`` (advance_time semantics,
zero-time no-op, rhs-eval counting, reinit) onto the ported DOLFINx
``LLG``/``Field`` stack, and adds the new contracts from
``dev/dolfinx/porting_map.md`` ("Field `xxx` consumers"): the SciPy driver
must seed and write back the ODE state through the explicit
component-blocked ``xxx`` ordering, not the raw backend dof order.
"""

import sys

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI
from scipy.integrate import ode

import finmag.util.consts as consts
from finmag.energies import Zeeman
from finmag.physics.llg import LLG
from finmag.drivers.scipy_integrator import ScipyIntegrator
from finmag.drivers.llg_integrator import llg_integrator, SundialsIntegrator

EPSILON = 1e-15


def _spaces(domain):
    S1 = fem.functionspace(domain, ("Lagrange", 1))
    S3 = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    return S1, S3


def _macrospin_llg(m, Hz, alpha=0.1, Ms=8.6e5, do_precession=True):
    """Uniform-state LLG on a single-cell cube (every node identical)."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    S1, S3 = _spaces(domain)
    llg = LLG(S1, S3, do_precession=do_precession, unit_length=1e-9)
    llg.Ms = Ms
    llg.set_alpha(alpha)
    llg.set_m(tuple(m), normalise=True)
    llg.effective_field.add(Zeeman((0.0, 0.0, Hz), name="Zeeman"))
    return llg


def _nonuniform_llg(Hz=1.0e5, alpha=0.1, Ms=8.6e5):
    """Multi-vertex LLG with a position-dependent initial magnetisation."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    S1, S3 = _spaces(domain)
    llg = LLG(S1, S3, do_precession=True, unit_length=1e-9)
    llg.Ms = Ms
    llg.set_alpha(alpha)
    llg.set_m(lambda x: np.vstack((x[0] + 0.1, x[1] + 0.2, np.ones_like(x[0]))))
    llg.effective_field.add(Zeeman((0.0, 0.0, Hz), name="Zeeman"))
    return llg


# --------------------------------------------------------------------------
# import boundary
# --------------------------------------------------------------------------

def test_scipy_driver_does_not_load_legacy_dolfin_or_native():
    assert ScipyIntegrator.__module__ == "finmag.drivers.scipy_integrator"
    # Legacy dolfin must never load on the DOLFINx stack, regardless of backend.
    assert "dolfin" not in sys.modules
    # Task 20: this test module imports ``SundialsIntegrator`` at top level,
    # which runs the lazy availability probe. When the native extension is now
    # BUILT (DOLFINx env post-Task-20), that probe legitimately imports
    # ``finmag.native.sundials`` and leaves it loaded -- so the native-free
    # assertion only applies where the sundials backend is genuinely absent
    # (the probe rolls back its residue on failure). The scipy path itself never
    # imports native; that is pinned by the ScipyIntegrator.__module__ check
    # above and by test_import_boundary.py's plain-import contract. [Claude Opus 4.8]
    if SundialsIntegrator is None:
        assert not any(name.startswith("finmag.native") for name in sys.modules)


# --------------------------------------------------------------------------
# checklist item 2: stiff VODE/BDF probe (no LLG involved)
# --------------------------------------------------------------------------

def test_vode_bdf_stiff_probe_matches_analytic_solution():
    """A classic stiff linear ODE, solved with the pinned scipy VODE/BDF path.

    y' = lam*(y - cos(t)) - sin(t), whose exact solution is
    y(t) = cos(t) + (y0 - 1) * exp(lam * t). With lam very negative this is
    stiff (fast transient relaxing onto a slowly varying solution), which is
    exactly the regime the legacy Sundials/VODE "bdf" method path exists for.
    """
    lam = -1000.0
    y0 = 2.0

    def rhs(t, y):
        return [lam * (y[0] - np.cos(t)) - np.sin(t)]

    solver = ode(rhs, jac=None)
    solver.set_integrator("vode", method="bdf", rtol=1e-8, atol=1e-10, nsteps=100000)
    solver.set_initial_value([y0], 0.0)

    t1 = 1.0
    y1 = solver.integrate(t1)
    assert solver.successful()

    analytic = np.cos(t1) + (y0 - 1.0) * np.exp(lam * t1)
    assert y1[0] == pytest.approx(analytic, abs=1e-6)


# --------------------------------------------------------------------------
# checklist item 3: advance_time / tolerance interface preserved
# --------------------------------------------------------------------------

def test_constructor_signature_and_defaults_preserved():
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    integrator = ScipyIntegrator(llg, llg.m_field)
    assert integrator.cur_t == 0.0
    assert integrator.n_rhs_evals == 0
    assert integrator.tablewriter is None


def test_advance_time_zero_first_is_a_no_op():
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    integrator = ScipyIntegrator(llg, llg.m_field)
    integrator.advance_time(0)
    assert integrator.cur_t == 0.0
    assert integrator.n_rhs_evals == 0
    # Subsequent non-zero advances still work after the zero no-op.
    integrator.advance_time(1e-12)
    assert integrator.cur_t == 1e-12


def test_advance_time_moves_cur_t_and_updates_m_field():
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5, alpha=0.1)
    integrator = ScipyIntegrator(llg, llg.m_field, reltol=1e-8, abstol=1e-10)
    integrator.advance_time(1e-11)
    assert integrator.cur_t == 1e-11
    # Magnetisation should have moved away from the exact initial state
    # (every vertex is identical in this uniform macrospin fixture).
    m = llg.m_field.get_ordered_numpy_array_xxx().reshape((3, -1))
    assert not np.allclose(m[:, 0], [1.0, 0.0, 0.0])


def test_n_rhs_evals_increases_after_advance():
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    integrator = ScipyIntegrator(llg, llg.m_field)
    assert integrator.n_rhs_evals == 0
    integrator.advance_time(1e-11)
    assert integrator.n_rhs_evals > 0


# --------------------------------------------------------------------------
# macrospin relaxation towards the field (ported from legacy scipy smoke use)
# --------------------------------------------------------------------------

def test_macrospin_relaxes_towards_field_within_physical_time():
    Hz = 1.0e5
    alpha = 0.5
    theta = 0.3
    m0 = (np.sin(theta), 0.0, np.cos(theta))
    llg = _macrospin_llg(m0, Hz, alpha=alpha)
    integrator = ScipyIntegrator(llg, llg.m_field, reltol=1e-8, abstol=1e-10)

    gamma_LL = consts.gamma / (1.0 + alpha * alpha)
    damping_time = 1.0 / (alpha * gamma_LL * Hz)
    integrator.advance_time(20 * damping_time)

    m = llg.m_field.get_ordered_numpy_array_xxx().reshape((3, -1))
    assert m[2, 0] == pytest.approx(1.0, abs=1e-3)
    assert np.linalg.norm(m[:, 0]) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# checklist item: explicit xxx state routing (the resolved raw/xxx ambiguity)
# --------------------------------------------------------------------------

def test_orderings_differ_on_this_mesh():
    """Sanity check: raw backend order and xxx order are genuinely different
    for the fixture mesh used below, so the following tests are meaningful."""
    llg = _nonuniform_llg()
    raw = llg.m_field.as_array()
    xxx = llg.m_field.get_ordered_numpy_array_xxx()
    assert raw.shape == xxx.shape
    assert not np.allclose(np.sort(raw), np.sort(xxx)) or not np.allclose(raw, xxx)
    assert not np.allclose(raw, xxx)


def test_scipy_integrator_seeds_ode_state_with_xxx_not_raw_array():
    llg = _nonuniform_llg()
    integrator = ScipyIntegrator(llg, llg.m_field)
    expected_xxx = llg.m_field.get_ordered_numpy_array_xxx()
    raw = llg.m_field.as_array()

    assert np.allclose(integrator.ode.y, expected_xxx)
    # Guard against silently reintroducing the legacy raw-order seed bug.
    assert not np.allclose(integrator.ode.y, raw)


def test_feeding_raw_order_into_solve_for_gives_wrong_physics():
    """Demonstrates the bug the driver must not reintroduce: interpreting the
    raw backend-order array as if it were the xxx state vector scrambles the
    magnetisation and changes the computed dm/dt relative to the correctly
    xxx-ordered state."""
    llg = _nonuniform_llg()
    correct_xxx = llg.m_field.get_ordered_numpy_array_xxx()
    raw = llg.m_field.as_array()
    assert not np.allclose(correct_xxx, raw)

    dmdt_correct = llg.solve_for(correct_xxx, 0.0).copy()
    dmdt_wrong = llg.solve_for(raw, 0.0).copy()

    assert not np.allclose(dmdt_correct, dmdt_wrong)


def test_advance_time_writes_back_through_xxx_ordering():
    llg = _nonuniform_llg(alpha=0.1)
    integrator = ScipyIntegrator(llg, llg.m_field, reltol=1e-8, abstol=1e-10)
    integrator.advance_time(1e-13)
    new_state = integrator.ode.y
    assert np.allclose(llg.m_field.get_ordered_numpy_array_xxx(), new_state)


# --------------------------------------------------------------------------
# checklist item 5: reject backward / unsuccessful integration explicitly
# --------------------------------------------------------------------------

def test_advance_time_backward_raises_value_error():
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    integrator = ScipyIntegrator(llg, llg.m_field)
    integrator.advance_time(1e-11)
    with pytest.raises(ValueError):
        integrator.advance_time(0.5e-11)


def test_unsuccessful_integration_raises_runtime_error():
    # nsteps=1 makes VODE give up almost immediately on a nontrivial problem,
    # which must surface as a real exception rather than a bare assert (so it
    # is visible under `python -O`, where `assert` is compiled out).
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5, alpha=0.1)
    integrator = ScipyIntegrator(llg, llg.m_field, nsteps=1)
    # scipy's vode itself prints/warns "Excess work done..." on this failure;
    # that expected warning is asserted here rather than left as test noise.
    with pytest.warns(UserWarning, match="Excess work"):
        with pytest.raises(RuntimeError):
            integrator.advance_time(1e-6)


# --------------------------------------------------------------------------
# checklist item 4: real reinitialization from the current field state
# --------------------------------------------------------------------------

def _advance_pair(alpha=0.1, t1=5e-11):
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5, alpha=alpha)
    integrator = ScipyIntegrator(llg, llg.m_field, reltol=1e-9, abstol=1e-11)
    integrator.advance_time(t1)
    return llg, integrator


def test_reinit_makes_external_field_modification_take_effect():
    external_state = np.array([0.0, 1.0, 0.0])
    tiny_dt = 1e-16

    # Without reinit: an external modification to m_field is silently
    # overwritten by the integrator's own internal (stale) state on the next
    # advance_time call.
    llg_a, integrator_a = _advance_pair()
    baseline = llg_a.m_field.get_ordered_numpy_array_xxx().reshape((3, -1))[:, 0].copy()
    llg_a.m_field.set(tuple(external_state), normalised=True)
    integrator_a.advance_time(integrator_a.cur_t + tiny_dt)
    result_without_reinit = llg_a.m_field.get_ordered_numpy_array_xxx().reshape((3, -1))[:, 0]
    assert not np.allclose(result_without_reinit, external_state, atol=1e-3)
    assert np.allclose(result_without_reinit, baseline, atol=1e-3)

    # With reinit: the same external modification is preserved (up to the
    # negligible physical evolution over tiny_dt).
    llg_b, integrator_b = _advance_pair()
    llg_b.m_field.set(tuple(external_state), normalised=True)
    integrator_b.reinit()
    integrator_b.advance_time(integrator_b.cur_t + tiny_dt)
    result_with_reinit = llg_b.m_field.get_ordered_numpy_array_xxx().reshape((3, -1))[:, 0]
    assert np.allclose(result_with_reinit, external_state, atol=1e-3)


def test_reinit_preserves_cur_t():
    llg, integrator = _advance_pair(t1=3e-11)
    t_before = integrator.cur_t
    integrator.reinit()
    assert integrator.cur_t == t_before


# --------------------------------------------------------------------------
# checklist item 6: scipy is the default/supported DOLFINx backend
# --------------------------------------------------------------------------

def test_llg_integrator_default_backend_is_scipy():
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    integrator = llg_integrator(llg, llg.m_field)
    assert isinstance(integrator, ScipyIntegrator)


def test_llg_integrator_explicit_scipy_backend():
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    integrator = llg_integrator(llg, llg.m_field, backend="scipy")
    assert isinstance(integrator, ScipyIntegrator)


def test_llg_integrator_sundials_backend_raises_by_name():
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    if SundialsIntegrator is not None:
        pytest.skip("native sundials extension is available in this environment")
    with pytest.raises(ImportError, match="sundials"):
        llg_integrator(llg, llg.m_field, backend="sundials")


def test_llg_integrator_unknown_backend_raises_value_error():
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    with pytest.raises(ValueError):
        llg_integrator(llg, llg.m_field, backend="bogus")
