"""Task 20: native Sundials/CVODE backend on the DOLFINx stack.

Formerly ``src/finmag/tests/test_sundials_driver_dolfinx.py``; renamed onto the
sibling-free canonical name (canonical-test-paths move, 2026-07-27).
Ancestors REMOVED (every test function accounted for by a named port
function below): ``util/ode/tests/test_sundials_ode.py``,
``drivers/tests/sundials_reinit_test.py``.
Ancestors RETAINED: ``drivers/tests/sundials_nsteps_test.py`` and
``util/ode/tests/test_sundials_stiff_ode.py`` -- the header accounts several of
their functions as "covered-elsewhere: sibling file, still passing", i.e. the
ancestor file ITSELF is the covering coverage, so it must stay.

This is the gate for the dolfin-free native Sundials/CVODE port
(``dolfinx-src-sundials-pytest``). It exercises four layers:

1. the native CVODE wrapper directly (``finmag.native.sundials``), porting the
   untouched legacy oracle invariants from
   ``src/finmag/util/ode/tests/test_sundials_ode.py`` (simple exponential ODE
   under ADAMS/functional, BDF/functional, BDF/Newton+diag, and BDF/Newton +
   SPGMR with a Jacobian-times-vector callback), plus the error / callback
   exception invariants;

2. the ``SundialsIntegrator`` driver wired against the ported deterministic
   ``LLG`` (analytic macrospin relaxation, xxx-ordered state, the reinit
   counter-reset and max_steps/advance_steps semantics ported from
   ``src/finmag/drivers/tests/sundials_reinit_test.py`` /
   ``sundials_nsteps_test.py``);

3. the analytic Jacobian-times-vector hook (``LLG.sundials_jtimes``) checked
   against a finite-difference of the right-hand side; and

4. cross-backend agreement: the same Task 9/10 core workflow
   (Exchange + Zeeman + UniaxialAnisotropy on a 3D box, ``run_until(1e-11)``)
   on the SciPy vs native Sundials backends must agree on m(t_end) within a
   tolerance justified from the integrators' rtol/atol, and a barmini-class
   ``Simulation`` advance mirroring the pixi ``barmini-smoke`` acceptance slice.

Every test that needs the native extension skips cleanly when it is
unavailable (mirroring the Task 10 skip guards), so this file also collects on
the legacy stack without the native build present.

MASTER->PORT MAPPING-HEADER (BUCKET-B accounting)
---------------------------------------------------
Four master ancestors feed this port. Two of them (``sundials_reinit_test.py``
and ``test_sundials_ode.py``) cannot even be collected under the DOLFINx/
python3 stack any more -- they (or a module they import) still do
``import dolfin`` -- so this file is their *sole* surviving coverage. The
other two (``sundials_nsteps_test.py`` and ``test_sundials_stiff_ode.py``) are
dolfin-free and still collect and pass, unmodified in spirit, as siblings
under the current stack; they are left in place as "covered-elsewhere" rather
than duplicated here.

``src/finmag/util/ode/tests/test_sundials_ode.py`` (class ``OdeSundialsTests``,
blocked from collecting here: ``finmag.util.helpers`` -> ``import dolfin``):

| master fn                              | status                                                                 |
|-----------------------------------------|------------------------------------------------------------------------|
| ``test_errors``                         | covered: ``test_native_uninitialised_advance_raises_runtime_error``   |
| ``test_simple_1d_scipy``                | covered-elsewhere: ``drivers/tests/test_scipy.py::test_vode_bdf_stiff_probe_matches_analytic_solution`` (different ODE -- stiff forced linear decay vs plain exponential growth -- same oracle concern: bare ``scipy.integrate.ode``/VODE/BDF matches an analytic solution) |
| ``test_simple_1d`` (Adams+BDF/functional) | covered: ``test_native_simple_ode_adams_and_bdf_functional``        |
| ``test_simple_1d_diag`` (BDF/Newton+diag) | covered: ``test_native_simple_ode_bdf_newton_diag``                 |
| ``test_stiff_sp_gmr`` (BDF/Newton+SPGMR/jtimes) | covered, **behaviour tightened**: ``test_native_simple_ode_bdf_newton_spgmr_jtimes``. Master's ``jtimes`` callback never wrote ``Jv`` (returned 0 having only computed nothing), so it exercised the SPGMR call path without checking the Jacobian value; the port's ``jtimes`` sets ``Jv[:] = 0.5 * v`` (the true Jacobian of ``0.5 y``), so the same <1e-6 tolerance now also certifies the Jacobian-vector product itself. Flagged inline. |
| ``test_jtimes_ex``                      | covered: ``test_native_jtimes_exception_propagates``                  |
| ``init_simple_test``/``run_simple_test`` (helpers) | covered: folded into ``_run_simple_exponential`` helper here    |

``src/finmag/drivers/tests/sundials_reinit_test.py`` (blocked from collecting
here: imports ``finmag.tests.jacobean.domain_wall_cobalt`` -> ``import dolfin``):

| master fn                                       | status                                                        |
|--------------------------------------------------|----------------------------------------------------------------|
| ``test_reinit_resets_num_rhs_eval_counter``       | covered, fixture simplified: ``test_sundials_reinit_resets_num_rhs_eval_counter``. Master iterates the ``domain_wall_cobalt`` fixture across ``method in {"bdf_diag", "adams"}`` (the ``"adams"`` case is actually run twice in master, likely a copy/paste artefact -- there is no assertion there beyond the one exercised below); the port instead drives the default macrospin fixture through the driver's default ``bdf_gmres_prec_id`` method. The single master-level invariant that survives across every variant -- ``reinit()`` zeroes ``n_rhs_evals`` and integration remains usable afterwards -- is kept at master's exact assertion (``== 0``), no tolerance loosened. |
| ``run_test`` (helper)                             | not a test; folded into the port's use of ``llg_integrator`` + ``_macrospin_llg`` directly |

``src/finmag/drivers/tests/sundials_nsteps_test.py`` (dolfin-free via
``finmag.example.barmini`` -> DOLFINx ``bar.py``; collects and passes
unmodified as a sibling under ``pixi run -e dolfinx pytest``, confirmed
5 passed alongside the stiff-ODE file in the same run):

| master fn                              | status                                                                   |
|------------------------------------------|----------------------------------------------------------------------------|
| ``test_integrator_get_set_max_steps``     | covered, both directly (``test_sundials_max_steps_get_set``, simplified macrospin fixture) and covered-elsewhere (sibling file, unmodified, still passing) |
| ``test_integrator_stats`` (all stats keys == 0 pre-integration) | covered-elsewhere only: sibling file, unmodified, still passing; not duplicated here |
| ``test_integrator_n_steps_only`` (nsteps progression, ``cur_t==tcur``, and ``tcur==hlast`` after 1 step) | covered, directly for nsteps/``cur_t==tcur`` (``test_sundials_advance_steps_counts_internal_steps``); the ``tcur==hlast`` after-one-step check is covered-elsewhere only (sibling file, unmodified, still passing) |

``src/finmag/util/ode/tests/test_sundials_stiff_ode.py`` (Robertson stiff-ODE
Jacobian-orientation convention; dolfin-free, collects and passes unmodified
in spirit -- already modernised in-place by prior work with real assertions
added and the two pathological transposed-Jacobian cases marked
``xfail(strict=True)`` -- as a sibling under ``pixi run -e dolfinx pytest``):

| master fn                          | status                                                                             |
|--------------------------------------|--------------------------------------------------------------------------------------|
| ``test_robertson_scipy``             | covered-elsewhere: sibling file (modernised, passing)                                |
| ``test_robertson_scipy_transposed``  | covered-elsewhere: sibling file, renamed ``..._fails_with_excess_work``, ``xfail(strict=True)`` (divergence visible) |
| ``test_robertson_sundials``          | covered-elsewhere: sibling file (modernised, passing, now with added value/step assertions master lacked) |
| ``test_robertson_sundials_transposed`` | covered-elsewhere: sibling file, renamed ``..._fails_with_excess_work``, ``xfail(strict=True)`` (divergence visible) |

No genuinely-dropped coverage was found: every master assertion is either
ported directly below or still exercised, passing, by an unmodified or
modernised sibling file under the current DOLFINx/python3 stack. Nothing was
silently loosened; the one behaviour tightening (real vs. no-op ``jtimes`` in
the SPGMR case) and the two pre-existing pathological-Jacobian ``xfail``
markers are called out explicitly above and inline.
"""

import math

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

import finmag.util.consts as consts
from finmag.energies import Exchange, UniaxialAnisotropy, Zeeman
from finmag.physics.llg import LLG
from finmag.sim.sim import Simulation
from finmag.drivers.scipy_integrator import ScipyIntegrator
from finmag.drivers.llg_integrator import llg_integrator, SundialsIntegrator

# The native module is imported lazily; a failed probe leaves SundialsIntegrator
# as None (unported/legacy-absent env) and every native test below skips.
try:
    import finmag.native.sundials as native_sundials
except Exception:  # pragma: no cover - exercised only where the build is absent
    native_sundials = None

requires_sundials = pytest.mark.skipif(
    SundialsIntegrator is None or native_sundials is None,
    reason="native sundials extension is not available in this environment",
)


# --------------------------------------------------------------------------
# fixtures (shared with the SciPy driver gate's construction conventions)
# --------------------------------------------------------------------------

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


def _core_workflow_sim(name, backend):
    """The Task 9/10 core workflow: 3D box + Exchange/Zeeman/UniaxialAnisotropy."""
    box = mesh.create_box(
        MPI.COMM_WORLD,
        [(0.0, 0.0, 0.0), (5.0, 5.0, 5.0)],
        [2, 2, 2],
        mesh.CellType.tetrahedron,
    )
    sim = Simulation(box, 8.6e5, unit_length=1e-9, name=name,
                     integrator_backend=backend)
    sim.alpha = 0.5
    sim.set_m((1.0, 0.0, 0.0))
    sim.add(Exchange(13.0e-12))
    sim.add(Zeeman((0.0, 0.0, 1e6)))
    sim.add(UniaxialAnisotropy(1e5, (0.0, 0.0, 1.0)))
    sim.set_tol(reltol=1e-8, abstol=1e-10)
    return sim


# --------------------------------------------------------------------------
# layer 1: native CVODE wrapper -- ported legacy oracle invariants
# (src/finmag/util/ode/tests/test_sundials_ode.py, untouched history)
# --------------------------------------------------------------------------

def _run_simple_exponential(integrator):
    """dy/dt = 0.5 y, y(0)=1 -> y(t)=exp(0.5 t), matched to <1e-6."""
    def rhs(t, y, ydot):
        ydot[:] = 0.5 * y
        return 0

    integrator.init(rhs, 0, np.array([1.0]))
    integrator.set_scalar_tolerances(1e-9, 1e-9)

    yout = np.zeros(1)
    ts = np.linspace(0.001, 3, 100)
    ys = np.zeros((100, 1))
    for i, t in enumerate(ts):
        integrator.advance_time(t, yout)
        ys[i] = yout.copy()
    ref = np.array([[math.exp(0.5 * t)] for t in ts])
    assert np.max(np.abs(ys - ref)) < 1e-6


@requires_sundials
def test_native_reports_sundials_7():
    assert native_sundials.get_sundials_version().startswith("7")


@requires_sundials
def test_native_simple_ode_adams_and_bdf_functional():
    _run_simple_exponential(
        native_sundials.cvode(native_sundials.CV_ADAMS,
                              native_sundials.CV_FUNCTIONAL))
    _run_simple_exponential(
        native_sundials.cvode(native_sundials.CV_BDF,
                              native_sundials.CV_FUNCTIONAL))


@requires_sundials
def test_native_simple_ode_bdf_newton_diag():
    integrator = native_sundials.cvode(native_sundials.CV_BDF,
                                       native_sundials.CV_NEWTON)

    def rhs(t, y, ydot):
        ydot[:] = 0.5 * y
        return 0

    integrator.init(rhs, 0, np.array([1.0]))
    integrator.set_scalar_tolerances(1e-9, 1e-9)
    integrator.set_linear_solver_diag()

    yout = np.zeros(1)
    ts = np.linspace(0.001, 3, 100)
    ys = np.zeros((100, 1))
    for i, t in enumerate(ts):
        integrator.advance_time(t, yout)
        ys[i] = yout.copy()
    ref = np.array([[math.exp(0.5 * t)] for t in ts])
    assert np.max(np.abs(ys - ref)) < 1e-6


@requires_sundials
def test_native_simple_ode_bdf_newton_spgmr_jtimes():
    """Ported from ``test_sundials_ode.py::test_stiff_sp_gmr``, tightened.

    Master's ``jtimes`` callback never wrote ``Jv`` at all (just ``return 0``)
    -- it only exercised the SPGMR call path, not the Jacobian-vector value.
    This port's ``jtimes`` sets the true ``Jv = 0.5 * v``, so the same <1e-6
    tolerance now also certifies the analytic Jacobian-vector product.
    Behaviour change flagged per the BUCKET-B mapping-header above.
    """
    integrator = native_sundials.cvode(native_sundials.CV_BDF,
                                       native_sundials.CV_NEWTON)

    def rhs(t, y, ydot):
        ydot[:] = 0.5 * y
        return 0

    def jtimes(v, Jv, t, y, fy, tmp):
        # Exact Jacobian of 0.5 y is 0.5 I; J v = 0.5 v.
        Jv[:] = 0.5 * v
        return 0

    integrator.init(rhs, 0, np.array([1.0]))
    integrator.set_scalar_tolerances(1e-9, 1e-9)
    integrator.set_linear_solver_sp_gmr(native_sundials.PREC_NONE)
    integrator.set_spils_jac_times_vec_fn(jtimes)

    yout = np.zeros(1)
    ts = np.linspace(0.001, 3, 100)
    ys = np.zeros((100, 1))
    for i, t in enumerate(ts):
        integrator.advance_time(t, yout)
        ys[i] = yout.copy()
    ref = np.array([[math.exp(0.5 * t)] for t in ts])
    assert np.max(np.abs(ys - ref)) < 1e-6


@requires_sundials
def test_native_uninitialised_advance_raises_runtime_error():
    integrator = native_sundials.cvode(native_sundials.CV_ADAMS,
                                       native_sundials.CV_FUNCTIONAL)
    with pytest.raises(RuntimeError):
        integrator.advance_time(1, np.zeros((5,)))


@requires_sundials
def test_native_jtimes_exception_propagates():
    class MyException(Exception):
        pass

    integrator = native_sundials.cvode(native_sundials.CV_BDF,
                                       native_sundials.CV_NEWTON)

    def rhs(t, y, ydot):
        ydot[:] = 0.5 * y
        return 0

    def jtimes(v, Jv, t, y, fy, tmp):
        raise MyException()

    integrator.init(rhs, 0, np.array([1.0]))
    integrator.set_scalar_tolerances(1e-9, 1e-9)
    integrator.set_linear_solver_sp_gmr(native_sundials.PREC_NONE)
    integrator.set_spils_jac_times_vec_fn(jtimes)
    with pytest.raises(MyException):
        integrator.advance_time(1, np.zeros(1))


# --------------------------------------------------------------------------
# layer 2: SundialsIntegrator driver wired against the ported LLG
# --------------------------------------------------------------------------

@requires_sundials
def test_llg_integrator_sundials_backend_available():
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    integrator = llg_integrator(llg, llg.m_field, backend="sundials")
    assert isinstance(integrator, SundialsIntegrator)
    assert integrator.cur_t == 0.0


@requires_sundials
def test_llg_integrator_default_backend_is_sundials_and_advances():
    """Task 20 fix round 1: ``llg_integrator``'s ``backend`` default was
    flipped back from ``"scipy"`` to ``"sundials"``, restoring the legacy
    default semantics. A bare call with no explicit ``backend=`` must both
    construct a ``SundialsIntegrator`` (not just raise/skip) and be able to
    actually advance time -- construction alone would not catch a default
    that resolves to a broken or half-wired backend. [Claude Sonnet 5]
    """
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    integrator = llg_integrator(llg, llg.m_field)
    assert isinstance(integrator, SundialsIntegrator)
    assert integrator.advance_time(1e-12) is True
    assert integrator.cur_t == 1e-12


@requires_sundials
def test_sundials_advance_time_zero_first_is_a_no_op():
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    integrator = llg_integrator(llg, llg.m_field, backend="sundials")
    assert integrator.advance_time(0) is True
    assert integrator.cur_t == 0.0
    assert integrator.advance_time(1e-12) is True
    assert integrator.cur_t == 1e-12


@requires_sundials
def test_sundials_advance_time_backward_raises_runtime_error():
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    integrator = llg_integrator(llg, llg.m_field, backend="sundials")
    integrator.advance_time(1e-12)
    with pytest.raises(RuntimeError):
        integrator.advance_time(5e-13)


@requires_sundials
def test_sundials_macrospin_relaxes_towards_field_within_physical_time():
    """Analytic macrospin: with damping the moment aligns with the field.

    Closed-form endpoint for a macrospin in a field along +z: m -> +z_hat as
    t -> inf, with damping time 1/(alpha*gamma_LL*H). This is the same closed
    form the SciPy gate pins, now driven through native CVODE. [Claude Opus 4.8]
    """
    Hz = 1.0e5
    alpha = 0.5
    theta = 0.3
    m0 = (np.sin(theta), 0.0, np.cos(theta))
    llg = _macrospin_llg(m0, Hz, alpha=alpha)
    integrator = llg_integrator(llg, llg.m_field, backend="sundials",
                                reltol=1e-8, abstol=1e-10)

    gamma_LL = consts.gamma / (1.0 + alpha * alpha)
    damping_time = 1.0 / (alpha * gamma_LL * Hz)
    integrator.advance_time(20 * damping_time)

    m = llg.m_field.get_ordered_numpy_array_xxx().reshape((3, -1))
    assert m[2, 0] == pytest.approx(1.0, abs=1e-3)
    assert np.linalg.norm(m[:, 0]) == pytest.approx(1.0, abs=1e-6)


@requires_sundials
def test_sundials_reinit_resets_num_rhs_eval_counter():
    """Ported from sundials_reinit_test.py: reinit() zeroes the rhs counter.

    Master's ``sundials_reinit_test.py`` cannot even be collected under this
    stack any more (it imports the dolfin-only ``domain_wall_cobalt``
    fixture); this test is that master's sole surviving coverage. The
    domain-wall fixture and its ``bdf_diag``/``adams`` method sweep are
    replaced by the default macrospin fixture and the driver's default
    method, but the master-level invariant -- ``reinit()`` zeroes
    ``n_rhs_evals`` and the integrator remains usable afterwards -- is kept
    at master's exact tolerance (``== 0``). See the mapping-header table in
    this module's docstring.
    """
    llg = _macrospin_llg((np.sin(0.3), 0.0, np.cos(0.3)), 1.0e5, alpha=0.1)
    integrator = llg_integrator(llg, llg.m_field, backend="sundials",
                                reltol=1e-8, abstol=1e-10)
    integrator.advance_time(1e-11)
    assert integrator.n_rhs_evals > 0
    integrator.reinit()
    assert integrator.n_rhs_evals == 0
    # integration still proceeds correctly after reinit
    integrator.advance_time(2e-11)
    assert integrator.cur_t == 2e-11


@requires_sundials
def test_sundials_max_steps_get_set():
    """Ported from sundials_nsteps_test.py: max_steps round-trips."""
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    integrator = llg_integrator(llg, llg.m_field, backend="sundials",
                                nsteps=10000)
    assert integrator.max_steps == 10000
    integrator.max_steps = 42
    assert integrator.max_steps == 42


@requires_sundials
def test_sundials_advance_steps_counts_internal_steps():
    """Ported from sundials_nsteps_test.py: advance_steps advances nsteps."""
    llg = _macrospin_llg((np.sin(0.3), 0.0, np.cos(0.3)), 1.0e5, alpha=0.1)
    integrator = llg_integrator(llg, llg.m_field, backend="sundials",
                                reltol=1e-8, abstol=1e-10)
    assert integrator.stats()['nsteps'] == 0
    integrator.advance_steps(1)
    assert integrator.stats()['nsteps'] == 1
    assert integrator.cur_t == integrator.stats()['tcur']
    integrator.advance_steps(2)
    assert integrator.stats()['nsteps'] == 3


# --------------------------------------------------------------------------
# layer 3: analytic Jacobian-times-vector hook vs finite difference
# --------------------------------------------------------------------------

@requires_sundials
def test_sundials_jtimes_matches_finite_difference():
    """LLG.sundials_jtimes must equal the directional derivative of the rhs.

    J(m,t) mp = d/da rhs(m + a mp)|_{a=0}, checked against a symmetric finite
    difference of solve_for. This validates the analytic Jacobian kernel used
    by the default bdf_gmres_prec_id path. [Claude Opus 4.8]
    """
    llg = _macrospin_llg((np.sin(0.4), 0.1, np.cos(0.4)), 1.0e5, alpha=0.3)
    m = llg.m_field.get_ordered_numpy_array_xxx().copy()
    rng = np.random.default_rng(0)
    mp = rng.standard_normal(m.shape)

    t = 0.0
    J_mp = np.zeros_like(m)
    llg.sundials_jtimes(mp.copy(), J_mp, t, m.copy(), np.zeros_like(m),
                        np.zeros_like(m))

    eps = 1e-6
    fwd = llg.solve_for(m + eps * mp, t).copy()
    bwd = llg.solve_for(m - eps * mp, t).copy()
    fd = (fwd - bwd) / (2 * eps)

    # restore m for cleanliness
    llg.set_m(m, normalise=False)

    scale = max(np.max(np.abs(fd)), 1.0)
    assert np.max(np.abs(J_mp - fd)) / scale < 1e-5


@requires_sundials
def test_sundials_psolve_is_identity_and_psetup_records_state():
    llg = _macrospin_llg((1.0, 0.0, 0.0), 1.0e5)
    m = llg.m_field.get_ordered_numpy_array_xxx().copy()
    r = np.arange(m.size, dtype=np.float64)
    z = np.zeros_like(r)
    rc = llg.sundials_psolve(0.0, m, m, r, z, 0.0, 0.0, 1, m)
    assert rc == 0
    assert np.array_equal(z, r)

    retval, jcur = llg.sundials_psetup(0.0, m, m, False, 0.0, m, m, m)
    assert retval == 0
    assert jcur is True
    retval, jcur = llg.sundials_psetup(0.0, m, m, True, 0.0, m, m, m)
    assert jcur is False


# --------------------------------------------------------------------------
# layer 4: cross-backend agreement + barmini-class advance
# --------------------------------------------------------------------------

@requires_sundials
def test_cross_backend_agreement_scipy_vs_sundials():
    """Same core workflow on scipy vs sundials must agree at t_end.

    Both integrators are driven at reltol=1e-8, abstol=1e-10. Each controls its
    local truncation error to that level; the cross-integrator difference in
    m(t_end) is dominated by the independent accumulation of that controlled
    error over the run, so agreement to abs=1e-6 (about 100x rtol, |m|~O(1)) is
    the physically justified bound rather than a hand-tuned fit. [Claude Opus 4.8]
    """
    sim_scipy = _core_workflow_sim("cross_scipy", "scipy")
    sim_scipy.run_until(1e-11)
    m_scipy = sim_scipy.m.copy()

    sim_sun = _core_workflow_sim("cross_sundials", "sundials")
    sim_sun.run_until(1e-11)
    m_sun = sim_sun.m.copy()

    assert sim_sun.integrator_backend == "sundials"
    assert isinstance(sim_sun.integrator, SundialsIntegrator)
    assert np.max(np.abs(m_scipy - m_sun)) < 1e-6


@requires_sundials
def test_barmini_class_sim_advances_on_sundials_default_path():
    """barmini-class Simulation advance on the native default CVODE path.

    Mirrors the pixi ``barmini-smoke`` acceptance slice (run a short physical
    time and confirm progress + unit norm), but on the direct-DOLFINx stack and
    exercising the default ``bdf_gmres_prec_id`` method (jtimes + identity
    preconditioner). [Claude Opus 4.8]
    """
    sim = _core_workflow_sim("barmini_sundials", "sundials")
    sim.run_until(1e-12)
    assert sim.t >= 1e-12
    m_nodal = sim.m.reshape((3, -1))
    max_dev = float(np.max(np.abs(1.0 - np.linalg.norm(m_nodal, axis=0))))
    assert max_dev < 1e-5


# --------------------------------------------------------------------------
# layer 5: backend-neutral Simulation.reset_time (SR1 P1.1)
# --------------------------------------------------------------------------
# ``Simulation.reset_time`` used to reseed the freshly built integrator by
# poking at ``integrator.ode`` (a ``scipy.integrate.ode`` attribute that only
# the SciPy driver has), so it raised ``AttributeError`` on the native Sundials
# backend. The clock origin is now handed to the driver through
# ``llg_integrator(..., t0=t0)``, which both backends accept. [Claude Opus 4.8]

@requires_sundials
def test_sim_reset_time_to_zero_on_sundials_backend():
    sim = _core_workflow_sim("reset_zero_sundials", "sundials")
    sim.run_until(1e-12)
    assert sim.t > 0.0
    m_before = sim.m.copy()

    sim.reset_time(0.0)

    assert sim.t == 0.0
    assert isinstance(sim.integrator, SundialsIntegrator)
    # Resetting the clock must not disturb the magnetisation.
    assert np.array_equal(sim.m, m_before)


@requires_sundials
def test_sim_reset_time_to_nonzero_on_sundials_backend():
    sim = _core_workflow_sim("reset_nonzero_sundials", "sundials")
    sim.run_until(1e-12)
    m_before = sim.m.copy()

    sim.reset_time(5e-12)

    assert sim.t == 5e-12
    assert isinstance(sim.integrator, SundialsIntegrator)
    assert np.array_equal(sim.m, m_before)


@requires_sundials
def test_sim_integrates_forward_after_nonzero_reset_on_sundials_backend():
    """A nonzero reset must leave a usable integrator, not just a clock value."""
    sim = _core_workflow_sim("reset_forward_sundials", "sundials")
    sim.run_until(1e-12)
    sim.reset_time(5e-12)
    m_after_reset = sim.m.copy()

    sim.run_until(6e-12)

    assert np.isclose(sim.t, 6e-12)
    # The extra 1e-12 s of relaxation towards +z must have moved m.
    assert not np.allclose(sim.m, m_after_reset)
    m_nodal = sim.m.reshape((3, -1))
    max_dev = float(np.max(np.abs(1.0 - np.linalg.norm(m_nodal, axis=0))))
    assert max_dev < 1e-5


@requires_sundials
def test_reset_then_advance_matches_continuous_twin_on_sundials():
    """Physical oracle: a reset must shift the clock, not the trajectory.

    The core-workflow right-hand side is autonomous (Exchange + a constant
    Zeeman field + uniaxial anisotropy; no time-dependent term), so integrating
    a duration ``dt`` from a reset origin ``t0`` must land on the same
    magnetisation as integrating the same ``dt`` continuously from the same
    state. That is what a clock-only ``reset_time`` means physically, and it is
    the assertion that would fail if ``reset_time`` reported the right
    ``sim.t`` while quietly disturbing the state or the integration interval.

    Both runs use the native Sundials backend (asserted, not assumed). The two
    legs differ only in CVODE's internal step-size/order history -- the reset
    run restarts it, the twin carries it over -- so the residual between them
    is local truncation error, and it must scale with the requested tolerance.
    Measured over this 2 ps leg, it does: max|m_reset - m_twin| is 4.8e-8 at
    reltol=1e-8, 1.1e-9 at 1e-10, 6.1e-13 at 1e-12. The test therefore runs at
    ``reltol=1e-10 / abstol=1e-12`` (tighter than the file's usual 1e-8/1e-10,
    to buy discriminating power) and bounds the residual at 1e-8 -- roughly an
    order of magnitude above the measured value, so it is not brittle, while
    still sitting ~7 orders below the 0.32 physical drift over the same leg
    (asserted, so the comparison cannot pass vacuously). A reset that dropped
    or corrupted the state would land at the drift scale and be independent of
    tolerance, so this bound separates the two cleanly. [Claude Opus 4.8]
    """
    t_common, t0_reset, dt = 1e-12, 5e-12, 2e-12

    sim_reset = _core_workflow_sim("reset_continuity_reset", "sundials")
    sim_reset.set_tol(reltol=1e-10, abstol=1e-12)
    sim_reset.run_until(t_common)
    m_common = sim_reset.m.copy()
    sim_reset.reset_time(t0_reset)
    sim_reset.run_until(t0_reset + dt)
    m_reset = sim_reset.m.copy()

    sim_twin = _core_workflow_sim("reset_continuity_twin", "sundials")
    sim_twin.set_tol(reltol=1e-10, abstol=1e-12)
    sim_twin.run_until(t_common)
    # The twin only means anything if the two runs agree at the branch point.
    assert np.max(np.abs(sim_twin.m - m_common)) < 1e-12
    sim_twin.run_until(t_common + dt)
    m_twin = sim_twin.m.copy()

    assert isinstance(sim_reset.integrator, SundialsIntegrator)
    assert isinstance(sim_twin.integrator, SundialsIntegrator)
    assert np.isclose(sim_reset.t, t0_reset + dt)

    # Non-vacuous: the shared 2 ps leg moves the magnetisation by ~0.32.
    assert np.max(np.abs(m_twin - m_common)) > 1e-2
    assert np.max(np.abs(m_reset - m_twin)) < 1e-8
