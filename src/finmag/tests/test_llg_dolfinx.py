"""Focused production tests for the direct DOLFINx deterministic LLG core."""

import json
import os
import sys
import types

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

import finmag.util.consts as consts
from finmag.energies import Exchange, Zeeman
from finmag.field import Field
from finmag.physics.llg import LLG

FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "llg_rhs_nonuniform.json"
)


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


def _nodal_dmdt(dmdt_flat):
    """Return the per-node dm/dt (3, N) from a flat xxx array."""
    return dmdt_flat.reshape((3, -1))


# --------------------------------------------------------------------------
# import boundary
# --------------------------------------------------------------------------

def test_ported_llg_does_not_load_legacy_dolfin_or_native():
    assert LLG.__module__ == "finmag.physics.llg"
    assert "dolfin" not in sys.modules
    assert not any(name.startswith("finmag.native") for name in sys.modules)


# --------------------------------------------------------------------------
# analytic macrospin: precession sign, damping direction, time scale
# --------------------------------------------------------------------------

def test_macrospin_rhs_matches_closed_form_ll():
    Hz = 1.0e5
    alpha = 0.1
    theta = 0.3
    m = (np.sin(theta), 0.0, np.cos(theta))
    llg = _macrospin_llg(m, Hz, alpha=alpha)

    dmdt = _nodal_dmdt(llg.solve(0.0))[:, 0]

    s, c = np.sin(theta), np.cos(theta)
    gamma_LL = consts.gamma / (1.0 + alpha * alpha)
    # precession -gamma_LL (m x H); damping -alpha gamma_LL m x (m x H)
    expected = np.array(
        [
            -alpha * gamma_LL * (c * s * Hz),          # damping only (x)
            gamma_LL * s * Hz,                         # precession (y)
            alpha * gamma_LL * (s * s * Hz),           # damping only (z)
        ]
    )
    assert np.allclose(dmdt, expected, rtol=1e-10, atol=0.0)


def test_precession_sign_is_negative_gyromagnetic():
    # m along +x, H along +z: precession dm/dt must point along +y.
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5, alpha=0.0)
    dmdt = _nodal_dmdt(llg.solve(0.0))[:, 0]
    assert dmdt[1] > 0.0
    assert abs(dmdt[0]) < 1e-3 and abs(dmdt[2]) < 1e-3


def test_damping_rotates_m_towards_field():
    # m tilted from +z toward +x, H along +z. Damping must reduce m_x and
    # increase m_z (rotate m toward H).
    theta = 0.5
    llg = _macrospin_llg((np.sin(theta), 0.0, np.cos(theta)), 1.0e5, alpha=0.2,
                         do_precession=False)
    dmdt = _nodal_dmdt(llg.solve(0.0))[:, 0]
    assert dmdt[0] < 0.0   # m_x decreasing
    assert dmdt[2] > 0.0   # m_z increasing
    assert abs(dmdt[1]) < 1e-3


def test_precession_rate_sets_physical_time_scale():
    # For m perpendicular to H the precession rate equals gamma_LL * H.
    Hz = 1.0e5
    alpha = 0.05
    llg = _macrospin_llg((1.0, 0.0, 0.0), Hz, alpha=alpha)
    dmdt = _nodal_dmdt(llg.solve(0.0))[:, 0]
    gamma_LL = consts.gamma / (1.0 + alpha * alpha)
    assert dmdt[1] == pytest.approx(gamma_LL * Hz, rel=1e-10)


def test_do_precession_false_disables_only_precession():
    theta = 0.4
    m = (np.sin(theta), 0.0, np.cos(theta))
    Hz, alpha = 1.0e5, 0.15
    with_prec = _nodal_dmdt(_macrospin_llg(m, Hz, alpha=alpha,
                                           do_precession=True).solve(0.0))[:, 0]
    no_prec = _nodal_dmdt(_macrospin_llg(m, Hz, alpha=alpha,
                                         do_precession=False).solve(0.0))[:, 0]
    # Damping components (x, z) unchanged; precession component (y) removed.
    assert no_prec[1] == pytest.approx(0.0, abs=1e-3)
    assert with_prec[1] != pytest.approx(0.0, abs=1.0)
    assert np.allclose(no_prec[[0, 2]], with_prec[[0, 2]], rtol=1e-12)


# --------------------------------------------------------------------------
# unit-length invariance under integration
# --------------------------------------------------------------------------

def _rk4_step(llg, y, t, dt):
    k1 = llg.solve_for(y, t)
    k2 = llg.solve_for(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = llg.solve_for(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = llg.solve_for(y + dt * k3, t + dt)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def test_unit_length_preserved_and_full_period_precession():
    # alpha = 0: pure precession of m ⟂ H returns to start after one period.
    Hz = 1.0e5
    llg = _macrospin_llg((1.0, 0.0, 0.0), Hz, alpha=0.0)
    gamma_LL = consts.gamma
    period = 2.0 * np.pi / (gamma_LL * Hz)

    y = llg.sundials_m.copy()
    steps = 400
    dt = period / steps
    t = 0.0
    for _ in range(steps):
        y = _rk4_step(llg, y, t, dt)
        t += dt
        norms = np.linalg.norm(y.reshape((3, -1)), axis=0)
        assert np.allclose(norms, 1.0, atol=1e-6)

    m_final = y.reshape((3, -1))[:, 0]
    assert np.allclose(m_final, (1.0, 0.0, 0.0), atol=2e-3)


# --------------------------------------------------------------------------
# effective-field registry drives the RHS (no bypass)
# --------------------------------------------------------------------------

def test_rhs_is_driven_by_full_effective_field_registry():
    m = (1.0, 0.0, 0.0)
    llg = _macrospin_llg(m, 1.0e5, alpha=0.1)
    dmdt_one = _nodal_dmdt(llg.solve(0.0))[:, 0].copy()

    # Adding a second interaction must change the RHS (it is not bypassed).
    llg.effective_field.add(Zeeman((0.0, 5.0e4, 0.0), name="Zeeman2"))
    dmdt_two = _nodal_dmdt(llg.solve(0.0))[:, 0]
    assert not np.allclose(dmdt_one, dmdt_two)

    llg.effective_field.remove("Zeeman2")
    dmdt_back = _nodal_dmdt(llg.solve(0.0))[:, 0]
    assert np.allclose(dmdt_one, dmdt_back)


# --------------------------------------------------------------------------
# pinning
# --------------------------------------------------------------------------

def test_pinned_nodes_have_zero_dmdt():
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5, alpha=0.1)
    n_nodes = llg.m_numpy.size // 3
    llg.pins = [0, 2]
    dmdt = _nodal_dmdt(llg.solve(0.0))
    assert np.allclose(dmdt[:, [0, 2]], 0.0)
    # An unpinned node still evolves.
    unpinned = [i for i in range(n_nodes) if i not in (0, 2)]
    assert np.linalg.norm(dmdt[:, unpinned]) > 0.0


def test_out_of_range_pins_raise():
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    n_nodes = llg.m_numpy.size // 3
    with pytest.raises(ValueError, match="range"):
        llg.pins = [n_nodes + 5]


# --------------------------------------------------------------------------
# solve_for / xxx state round trip
# --------------------------------------------------------------------------

def test_solve_for_sets_state_and_returns_matching_rhs():
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5, alpha=0.1)
    new_state = llg.sundials_m.copy()
    new_state.reshape((3, -1))[:] = np.array([[0.0], [0.0], [1.0]])
    dmdt = llg.solve_for(new_state, 0.0)
    # State was applied.
    assert np.allclose(llg.m_numpy.reshape((3, -1)), [[0.0], [0.0], [1.0]])
    # m parallel to H (z): no precession, no damping -> ~0 RHS.
    assert np.allclose(_nodal_dmdt(dmdt), 0.0, atol=1e-3)


# --------------------------------------------------------------------------
# deferred surfaces
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["use_slonczewski", "use_zhangli",
                                     "sundials_jtimes", "sundials_psetup",
                                     "sundials_psolve"])
def test_deferred_surfaces_raise_by_name(method):
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    with pytest.raises(NotImplementedError):
        getattr(llg, method)()


def test_multi_rank_state_paths_raise_serial_guard():
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    llg.comm = types.SimpleNamespace(size=2)
    with pytest.raises(NotImplementedError, match="multi-rank"):
        llg.solve(0.0)
    with pytest.raises(NotImplementedError, match="multi-rank"):
        llg.solve_for(llg.sundials_m, 0.0)


def test_spatially_varying_alpha_deferred():
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    with pytest.raises(NotImplementedError, match="alpha"):
        llg.set_alpha(np.array([0.1, 0.2, 0.3]))


# --------------------------------------------------------------------------
# nonuniform oracle fixture
# --------------------------------------------------------------------------

def _lexsort_rows(coords, *arrays):
    padded = np.zeros((coords.shape[0], 3))
    padded[:, : coords.shape[1]] = coords
    order = np.lexsort((padded[:, 2], padded[:, 1], padded[:, 0]))
    return (padded[order],) + tuple(a[order] for a in arrays)


def test_nonuniform_rhs_matches_legacy_oracle_fixture():
    fixture = json.load(open(FIXTURE))
    params = fixture["physical_parameters"]
    cells = fixture["mesh"]["parameters"]["cells"]
    x1 = fixture["mesh"]["parameters"]["x1"]

    domain = mesh.create_interval(MPI.COMM_WORLD, cells, [0.0, x1])
    S1, S3 = _spaces(domain)
    llg = LLG(S1, S3, do_precession=params["do_precession"]["value"],
              unit_length=params["unit_length"]["value"])
    llg.Ms = params["Ms"]["value"]
    llg.set_alpha(params["alpha"]["value"])
    llg.set_m(
        lambda x: np.vstack(
            (np.cos(0.4 * x[0]), np.sin(0.4 * x[0]),
             0.5 * np.ones(x.shape[1]))
        ),
        normalise=True,
    )
    llg.effective_field.add(Exchange(params["A"]["value"], name="Exchange"))
    llg.effective_field.add(Zeeman(tuple(params["H_zeeman"]["value"]),
                                   name="Zeeman"))

    dmdt = llg.solve(0.0)

    cm, m_vals = llg.m_field.coords_and_values()
    Hf = Field(S3)
    Hf.from_array(llg.effective_field.H_eff)
    _, H_vals = Hf.coords_and_values()
    _, dmdt_vals = llg._dmdt.coords_and_values()

    _, m_sorted, H_sorted, dmdt_sorted = _lexsort_rows(cm, m_vals, H_vals, dmdt_vals)

    by_name = {q["name"]: q for q in fixture["quantities"]}
    for name, got in (("m", m_sorted), ("effective_field", H_sorted),
                      ("dmdt", dmdt_sorted)):
        q = by_name[name]
        ref = np.array(q["values"])
        tol = q["tolerances"]
        assert np.allclose(got, ref, atol=tol["absolute"], rtol=tol["relative"]), (
            "quantity {} mismatch:\n got {}\n ref {}".format(name, got, ref)
        )
