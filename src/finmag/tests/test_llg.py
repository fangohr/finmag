"""Focused production tests for the direct DOLFINx deterministic LLG core.

This file now lives at its master path
``src/finmag/tests/test_llg.py`` (formerly
``src/finmag/tests/test_llg_dolfinx.py``), so
``git diff b5015c5a..HEAD -- src/finmag/tests/test_llg.py``
shows the port diff directly.
"""

import json
import os
import subprocess
import sys
import types
from pathlib import Path

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

SRC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SRC_ROOT.parent


def _run_isolated(code, cwd=None, check=True):
    # Same pattern as ``finmag.tests.test_example._run_isolated``: run in a
    # fresh child interpreter so ``sys.modules`` reflects only what this
    # process actually imported, immune to whole-suite import ordering.
    # [Claude Sonnet 5]
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


# ``finmag.util.helpers`` imports ``dolfin`` at module load and is therefore
# unusable under DOLFINx; ``components`` is reimplemented here byte-identically
# to the master helper (``vs.view().reshape((3, -1))``) per the ratified
# minimal-diff workaround. [Claude Opus 4.8]
def components(vs):
    return vs.view().reshape((3, -1))


# ==========================================================================
# MASTER-DERIVED TESTS (faithful transcription of
# ``b5015c5a:src/finmag/tests/test_llg.py``)
# ==========================================================================

def test_method_of_computing_the_average_matters():
    length = 20e-9  # m
    simplices = 10
    # dolfin.IntervalMesh(simplices, 0, length) -> dolfinx.mesh.create_interval
    mesh_ = mesh.create_interval(MPI.COMM_WORLD, simplices, [0.0, length])
    # dolfin.FunctionSpace / VectorFunctionSpace -> dolfinx.fem.functionspace
    S1 = fem.functionspace(mesh_, ("Lagrange", 1))
    S3 = fem.functionspace(mesh_, ("Lagrange", 1, (3,)))

    llg = LLG(S1, S3)
    # DOCUMENTED API CHANGE: the port's ``set_m`` no longer accepts legacy
    # string ``Expression`` components with keyword parameters (it raises
    # ``NotImplementedError``); the identical field is supplied as the
    # equivalent callable. ``L=length`` is captured as a Python closure. The
    # ``np.maximum(..., 0.0)`` guards ``sqrt`` against tiny negative round-off
    # at the two endpoint nodes where ``(2x-L)/L == ±1`` exactly; the master
    # string ``sqrt`` was evaluated on the same nodal values.
    L = length
    llg.set_m(
        lambda x: np.vstack((
            (2 * x[0] - L) / L,
            np.sqrt(np.maximum(1.0 - ((2 * x[0] - L) / L) ** 2, 0.0)),
            np.zeros_like(x[0]),
        ))
    )

    average1 = llg.m_average
    average2 = np.mean(components(llg.m_numpy), axis=1)
    diff = np.abs(average1 - average2)
    # master tolerance kept verbatim; measured DOLFINx diff.max() = 0.06902
    assert diff.max() > 5e-2


# ===== NEW under DOLFINx (no master ancestor) =====


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
    # Run in a subprocess: under whole-suite ordering, earlier tests
    # legitimately import legacy ``dolfin``/``finmag.native`` into THIS
    # process, which would make the same in-process assertion fail on a
    # pollution the ported ``LLG`` itself never caused. A fresh child
    # interpreter is order-immune while asserting the identical contract.
    # [Claude Sonnet 5]
    result = _run_isolated(
        """
import sys
from finmag.physics.llg import LLG

assert LLG.__module__ == "finmag.physics.llg"
assert "dolfin" not in sys.modules
assert not any(name.startswith("finmag.native") for name in sys.modules)
"""
    )
    assert result.returncode == 0, result.stderr


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

# Task 20: sundials_jtimes / sundials_psetup / sundials_psolve are no longer
# by-name deferrals -- the native Sundials/CVODE backend is ported, so these are
# real implementations (analytic Jacobian-times-vector + identity
# preconditioner). Their behaviour is validated in
# test_sundials_driver.py.
#
# Task 22: the spin-transfer-torque surfaces (``use_slonczewski``/
# ``use_zhangli``) are now ported as NumPy transcriptions of the native STT
# kernels; they activate the corresponding torque instead of raising. Full
# behavioural + oracle coverage lives in test_stt.py; this only asserts
# they no longer raise ``NotImplementedError`` by name and set the right flag.
# [Claude Opus 4.8]
def test_stt_surfaces_are_ported_not_deferred():
    llg = _macrospin_llg((0.6, 0.0, 0.8), 8.6e5)
    llg.use_slonczewski(1.0e12, 0.4, 2e-9, (0.0, 0.0, 1.0))
    assert llg.do_slonczewski is True and llg.do_zhangli is False

    llg = _macrospin_llg((0.6, 0.0, 0.8), 8.6e5)
    llg.use_zhangli(J_profile=(1.0e12, 0.0, 0.0), P=0.5, beta=0.02)
    assert llg.do_zhangli is True and llg.do_slonczewski is False


def test_sundials_cvode_hooks_are_ported_not_deferred():
    """The CVODE preconditioner/Jacobian hooks are implemented (Task 20)."""
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    m = llg.sundials_m.copy()
    # psolve is the identity preconditioner z = r.
    r = np.arange(m.size, dtype=np.float64)
    z = np.zeros_like(r)
    assert llg.sundials_psolve(0.0, m, m, r, z, 0.0, 0.0, 1, m) == 0
    assert np.array_equal(z, r)
    # psetup records the linearisation state and reports (retval, jcur).
    assert llg.sundials_psetup(0.0, m, m, False, 0.0, m, m, m) == (0, True)
    # jtimes returns a finite same-shape Jacobian-times-vector product.
    J_mp = np.zeros_like(m)
    assert llg.sundials_jtimes(m.copy(), J_mp, 0.0, m.copy(),
                               np.zeros_like(m), np.zeros_like(m)) == 0
    assert J_mp.shape == m.shape and np.all(np.isfinite(J_mp))


def test_multi_rank_state_paths_raise_serial_guard():
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    llg.comm = types.SimpleNamespace(size=2)
    with pytest.raises(NotImplementedError, match="multi-rank"):
        llg.solve(0.0)
    with pytest.raises(NotImplementedError, match="multi-rank"):
        llg.solve_for(llg.sundials_m, 0.0)


def test_spatially_varying_alpha_is_supported():
    """Task 16: spatially varying alpha is now supported (see
    test_variable_params.py, formerly test_variable_params_dolfinx.py, for the
    oracle-pinned behaviour). A per-node
    array sets the nodal alpha field; a scalar stays a float."""
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    n = llg._alpha_field.as_array().size
    llg.set_alpha(np.linspace(0.1, 0.3, n))
    assert isinstance(llg.alpha, np.ndarray)
    assert llg.alpha.shape == (n,)
    llg.set_alpha(0.2)
    assert llg.alpha == 0.2


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
    # H_eff is now component-blocked (``xxx``, Task 31); reconstruct the
    # Function by inverting that ordering, not the raw from_array path (which
    # would scramble components and silently mis-order this oracle comparison).
    Hf.set_with_ordered_numpy_array_xxx(llg.effective_field.H_eff)
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


# --------------------------------------------------------------------------
# LLG.M / M_average magnetisation-in-A/m accessors (SR1 P4-M, register D20)
#
# ``M`` is the magnetisation in A/m (``Ms(x) * m(x)`` per node, component-blocked
# ``xxx``); ``M_average`` is its Ms-weighted volume average ``(integral Ms*m dV)
# / (integral dV)``. This is the CORRECT physics; the frozen legacy oracle was
# demonstrably broken -- ``LLG.M`` read ``self.m`` which raised
# ``RuntimeError("DON'T USE llg.m UNTIL FURTHER NOTICE!!!!")``, and
# ``LLG.M_average`` computed ``m_average * volume_Ms / volume`` with
# ``volume_Ms`` and ``volume`` the *identical* integral, collapsing it to the
# dimensionless ``m_average`` (a unit bug). See the D20 divergence pin below.
# --------------------------------------------------------------------------

def _multinode_llg(Ms=8.6e5, m=(0.0, 0.0, 1.0), n=2, unit_length=1e-9):
    """A multi-vertex cube LLG with a constant (or callable) Ms and m."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
    S1, S3 = _spaces(domain)
    llg = LLG(S1, S3, unit_length=unit_length)
    llg.Ms = Ms
    llg.set_m(m, normalise=True)
    return llg


def test_M_uniform_constant_ms_has_length_ms_at_every_node():
    """Uniform m, constant Ms: |M| == Ms at every node, direction == m."""
    Ms = 8.6e5
    llg = _multinode_llg(Ms=Ms, m=(0.0, 0.0, 1.0))

    M_nodes = llg.M.reshape((3, -1))
    norms = np.linalg.norm(M_nodes, axis=0)
    assert np.allclose(norms, Ms, rtol=1e-12, atol=0.0)
    # component-blocked xxx: only the z block carries Ms
    assert np.allclose(M_nodes[2], Ms, rtol=1e-12, atol=0.0)
    assert np.allclose(M_nodes[0], 0.0, atol=1e-6)
    assert np.allclose(M_nodes[1], 0.0, atol=1e-6)


def test_M_average_uniform_constant_ms_equals_ms_times_m_average():
    """Constant Ms: M_average == Ms * m_average (vector), |M_average| == Ms."""
    Ms = 8.6e5
    llg = _multinode_llg(Ms=Ms, m=(0.3, 0.0, 0.4))  # normalised -> (0.6,0,0.8)

    expected = Ms * llg.m_average
    assert np.allclose(llg.M_average, expected, rtol=1e-12, atol=0.0)
    assert np.isclose(np.linalg.norm(llg.M_average), Ms, rtol=1e-12)


def test_M_varying_ms_equals_per_node_ms_times_m_nonuniform():
    """Spatially varying Ms, NON-uniform m: M == Ms(x)*m(x) node-for-node.

    Anti-scramble: the non-uniform m means any component/node scramble in the
    ``xxx`` reshape would misalign the per-node product and fail. Verified on
    two independent orderings -- the flat ``xxx`` block layout and the
    coordinate (``coords_and_values``) vertex layout.
    """
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    S1, S3 = _spaces(domain)
    llg = LLG(S1, S3, unit_length=1e-9)
    llg.Ms = lambda x: 8.6e5 * (1.0 + 0.5 * x[0])
    llg.set_m(
        lambda x: np.vstack(
            (np.cos(1.7 * x[0]), np.sin(1.3 * x[1]), 0.5 + 0.2 * x[2])
        ),
        normalise=True,
    )

    Ms_node = llg._ms_nodal()                   # per-node Ms, coordinate order
    m_nodes = llg.m_numpy.reshape((3, -1))       # xxx, same node order
    expected = Ms_node * m_nodes                 # broadcast (N,) over (3,N)

    M_nodes = llg.M.reshape((3, -1))
    assert np.allclose(M_nodes, expected, rtol=1e-12, atol=0.0)

    # independent coordinate-path anti-scramble: rebuild M as an S3 Field and
    # compare vertex-by-vertex against Ms_node * m at matching coordinates.
    Mf = Field(S3)
    Mf.set_with_ordered_numpy_array_xxx(llg.M)
    _, Mvals = Mf.coords_and_values()
    _, mvals = llg.m_field.coords_and_values()
    assert np.allclose(Mvals, Ms_node[:, None] * mvals, rtol=1e-12, atol=0.0)


def test_M_array_is_xxx_component_blocked_matching_m_ordering():
    """The M array is Task-31 ``xxx`` component-blocked, matching m node-for-node.

    Dividing M's per-node blocks by the per-node Ms must recover ``m_numpy``
    exactly; a wrong ``(-1, 3)`` reshape would scramble nodes against components
    on this varying field and fail.
    """
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    S1, S3 = _spaces(domain)
    llg = LLG(S1, S3, unit_length=1e-9)
    llg.Ms = lambda x: 8.6e5 * (1.0 + 0.3 * x[1])
    llg.set_m(
        lambda x: np.vstack(
            (np.cos(2.1 * x[0]), np.sin(0.9 * x[2]), 0.4 + 0.3 * x[1])
        ),
        normalise=True,
    )

    Ms_node = llg._ms_nodal()
    recovered = llg.M.reshape((3, -1)) / Ms_node
    assert np.allclose(recovered, llg.m_numpy.reshape((3, -1)), rtol=1e-12)


def test_M_average_varying_ms_two_region_is_ms_weighted_volume_average():
    """Two equal-volume DG0 regions, uniform m: hand-computed Ms-weighted average.

    Interval [0, 2] split into cells [0,1] and [1,2] (equal volume 1) with
    per-cell Ms = [Ms1, Ms2] and uniform m = (0, 0, 1). The correct Ms-weighted
    volume average is analytically ((Ms1+Ms2)/2) * (0, 0, 1). Legacy's buggy
    M_average would have returned the dimensionless m_average = (0, 0, 1).
    """
    domain = mesh.create_interval(MPI.COMM_WORLD, 2, [0.0, 2.0])
    S1, S3 = _spaces(domain)
    llg = LLG(S1, S3, unit_length=1e-9)
    Ms1, Ms2 = 6.0e5, 9.0e5
    llg.Ms = np.array([Ms1, Ms2])
    llg.set_m((0.0, 0.0, 1.0), normalise=True)

    expected = np.array([0.0, 0.0, (Ms1 + Ms2) / 2.0])
    assert np.allclose(llg.M_average, expected, rtol=1e-12, atol=1.0)
    assert not np.allclose(llg.M_average, llg.m_average)


def test_M_average_varying_ms_matches_independent_assembly():
    """Varying Ms + non-uniform m: M_average matches an independent FEM oracle.

    Independently assembles ``(integral Ms*m_i dx) / (integral dx)`` and asserts
    the port's Ms-weighted volume average agrees, and that it is NOT the
    dimensionless ``m_average``.
    """
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    S1, S3 = _spaces(domain)
    llg = LLG(S1, S3, unit_length=1e-9)
    llg.Ms = lambda x: 8.6e5 * (1.0 + 0.5 * x[0])
    llg.set_m(
        lambda x: np.vstack(
            (np.cos(1.1 * x[0]), np.sin(0.7 * x[1]), 0.5 + 0.3 * x[2])
        ),
        normalise=True,
    )

    import ufl as _ufl

    Ms = llg._Ms_dg.f
    m = llg.m_field.f

    def _sc(form):
        return domain.comm.allreduce(
            fem.assemble_scalar(fem.form(form)), op=MPI.SUM
        )

    V = _sc(fem.Constant(domain, 1.0) * _ufl.dx)
    expected = np.array([_sc(Ms * m[i] * _ufl.dx) / V for i in range(3)])

    assert np.allclose(llg.M_average, expected, rtol=1e-12, atol=0.0)
    assert not np.allclose(llg.M_average, llg.m_average)


def test_M_average_diverges_from_legacy_dimensionless_m_average():
    """DIVERGENCE PIN (register D20): the port's M_average is the corrected
    Ms-weighted A/m value, NOT the legacy dimensionless m_average.

    The frozen legacy oracle ``b5015c5a:src/finmag/physics/llg.py`` computed::

        volume_Ms = df.assemble(self._Ms_dg * df.dx)
        volume    = df.assemble(self._Ms_dg * df.dx)   # identical integral
        return self.m_average * volume_Ms / volume     # -> m_average

    so ``volume_Ms == volume`` collapsed ``M_average`` to the dimensionless unit
    average ``m_average`` (a copy-paste unit bug; the docstring promised A/m).
    (Legacy ``LLG.M`` was separately broken: it read ``self.m``, which raised
    ``RuntimeError("DON'T USE llg.m UNTIL FURTHER NOTICE!!!!")``.)

    Under D20 the port returns the physically correct Ms-weighted average in
    A/m. This test records the resulting divergence from the buggy oracle so the
    behaviour change is documented, not silently erased. There is no consumer of
    ``M_average`` even in frozen master. See acceptance register row D20.
    """
    Ms = 8.6e5
    llg = _multinode_llg(Ms=Ms, m=(1.0, 0.0, 2.0))

    m_avg = llg.m_average          # dimensionless unit-vector average (legacy)
    M_avg = llg.M_average          # corrected Ms-weighted A/m average (port)

    # legacy would have returned m_avg; the port diverges by the Ms factor
    assert not np.allclose(M_avg, m_avg)
    assert np.allclose(M_avg, Ms * m_avg, rtol=1e-12, atol=0.0)
    # dimensionally A/m: |M_average| == Ms for this uniform state
    assert np.isclose(np.linalg.norm(M_avg), Ms, rtol=1e-12)
