"""Direct DOLFINx time-dependent Zeeman port (Task 15).

This file now lives at its master path ``src/finmag/energies/zeeman_test.py``
(formerly ``src/finmag/tests/test_timezeeman_dolfinx.py``), so
``git diff b5015c5a..HEAD -- src/finmag/energies/zeeman_test.py`` shows the
port diff directly. Because the port covers only 9 of master's 14 functions
here (4 more are covered in the static-Zeeman aggregate
``src/finmag/tests/test_energies.py``, formerly
``test_energies_dolfinx.py``), the one master function covered by NEITHER --
``test_compare_stray_field_of_sphere_with_dipolar_field`` -- is carried
verbatim at the bottom of this file under the ``NOT PORTED`` banner and
marked ``@pytest.mark.not_ported``.

Two clearly separated parts (three, counting the carried-verbatim tail),
following the SR1 P5.2 minimal-diff convention (exemplar:
``src/finmag/energies/demag/fk_demag_test.py``, formerly
``test_fk_demag_dolfinx.py``):

1. A MINIMAL-DIFF transcription of the subset of master's
   ``src/finmag/energies/zeeman_test.py`` (git ``b5015c5a``, 14 functions)
   that this file covers: the ``TimeZeeman``, ``DiscreteTimeZeeman``,
   ``OscillatingZeeman`` and ``DipolarField`` construction/behaviour tests --
   9 of the 14 master functions. Function names, ordering and assertion
   structure are kept identical to master; the only differences are
   (a) dolfin->dolfinx API changes (each annotated inline -- principally
   legacy's mutable-``.t``-attribute ``dolfin`` ``Expression`` becoming a
   plain Python ``field_function(t)`` callable, the ported input-contract
   deviation; see ``finmag/energies/zeeman.py`` module/class docstrings and
   ``transition-notes.org`` Task 15), (b) py2->py3 ``print``/``xrange``
   conversions, and (c) explanatory comments. Master's ``TOL = 1e-14`` is
   kept VERBATIM: measured DOLFINx max deviation across every ``TOL``-gated
   assertion in this file is ~8.9e-16 (see the probe note beside ``TOL``
   below), so no loosening was needed or done.

   ACCOUNTING for the remaining 5 master functions (so a reviewer can check
   off all 14 without re-deriving it):
   - ``test_compute_energy``, ``test_energy_density_function``,
     ``test_compute_energy_in_regions``, ``test_value_set_update`` -- these
     exercise the STATIC ``Zeeman`` class only (no time update at all) and
     are covered in ``src/finmag/tests/test_energies.py`` (formerly
     ``test_energies_dolfinx.py``) (bucket-B: aggregate
     mapping-header style, not a literal transcription; out of this file's
     edit boundary) by, respectively: ``test_zeeman_field_average_and_analytic_energy``
     (analytic ``mu0*Ms*dot(m,H)*volume``), ``test_zeeman_callable_set_value_keeps_live_function_and_density``
     (covers both ``energy_density()``/``energy_density_function()`` and
     ``set_value()``), and ``test_zeeman_compute_energy_preserves_restricted_measure_argument``
     (restricted ``dx`` sums to the unrestricted total, master's
     region-split contract).
   - ``test_compare_stray_field_of_sphere_with_dipolar_field`` -- NOT
     PORTED, and covered by NEITHER this file nor the static-Zeeman
     aggregate: already ``@pytest.mark.xfail(reason='dolfin 1.5')`` in
     master, and it depends on ``finmag.example.sphere_inside_airbox``,
     which under DOLFINx is a deferred surface/example that raises
     ``NotImplementedError`` by name
     (see ``test_example.py::test_deferred_surfaces_raise_named_not_implemented_error``
     and ``test_deferred_surfaces.py``, whose docstring confirms
     ``TimeZeemanPython``/``DipolarField`` themselves are ported while the
     airbox/surface machinery is not). Cannot be ported until that
     dependency lands -- so, rather than disappearing with master's file,
     it is CARRIED VERBATIM from ``b5015c5a`` at the bottom of this file
     under the ``NOT PORTED`` banner, marked ``@pytest.mark.not_ported``.
     Master's own ``@pytest.mark.xfail(reason='dolfin 1.5')`` already governs
     the outcome, so SR1 S0 (owner decision 2026-07-27) adds no further
     marker here; the ``dolfinx-src-timezeeman-pytest`` gate runs it
     unfiltered and reports it as xfailed (non-strict, per master's own
     marker), and it is also reported by the non-gating inventory lane.

   (An earlier pass's docstring described this file as "6/14 shared, 18
   new" -- that count was a literal function-NAME string match between the
   two files and undercounts real lineage by 3: ``test_time_zeeman_init``
   and ``test_oscillating_zeeman`` had simply been renamed in the port, and
   ``test_dipolar_field_class`` had no transcription at all despite being
   superseded by stronger tests below. All three are restored under their
   master names in part 1 below; the true split is 9 transcribed / 5
   not-covered-here, and 15 (not 18) genuinely-new tests in part 2.)

2. The sophisticated NEW-under-DOLFINx tests below the
   ``# ===== NEW under DOLFINx =====`` banner, which have no master
   ancestor (or, for the two ``DipolarField`` tests, materially stronger
   assertions than master's assertion-free smoke test). They validate:

   - ``TimeZeeman``/``DiscreteTimeZeeman`` update/switch-off semantics
     (held-value-between-updates, switch-off zeroes but keeps the
     interaction -- matching the Task 9 ``switch_off_H_ext`` precedent);
   - the related stale-energy defect -- ``DiscreteTimeZeeman`` used to
     never rebuild its cached energy form after an interval update, so
     ``compute_energy()`` went stale while ``compute_field()``/
     ``energy_density()`` stayed current -- discovered while building the
     oracle fixture below and now CORRECTED under acceptance register D3
     (approved 2026-07-23); see the D3 divergence pin below;
   - ``TimeZeemanPython``'s cached-spatial-pattern rescaling, and its
     by-name gate on the unported vector-valued ``time_fun`` branch
     (whole-branch review finding 1);
   - ``DipolarField``'s closed-form point-dipole field;
   - coordinate-ordered legacy oracle fixtures (``timezeeman_oracle.json``,
     regenerated by ``gen_timezeeman_oracle.py``) for one ``TimeZeeman``
     case (space+time varying) and one ``DiscreteTimeZeeman`` case (pinning
     the update-interval quirk bit-for-bit; its recorded energies are
     ~1e-37 J against an absolute tolerance of 1e-18, so it does not
     discriminate the D3 stale-energy correction and still passes
     unchanged);
   - the ``EffectiveField.add(..., with_time_update)`` auto-connection
     (Task 6), now exercised with the *real* ``TimeZeeman``/
     ``OscillatingZeeman`` classes via ``Simulation.add``/``run_until``
     (previously only a ``FakeTimeZeeman`` double).

[Claude Sonnet 5]
"""

import json
import os

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from finmag.energies import (
    DipolarField,
    DiscreteTimeZeeman,
    OscillatingZeeman,
    TimeZeeman,
    TimeZeemanPython,
    Zeeman,
)
from finmag.energies.energy_base import mu0
from finmag.field import Field
from finmag.sim.sim import Simulation


def _box(extent=1.0, n=3):
    return mesh.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (extent, extent, extent)],
        [n, n, n], mesh.CellType.tetrahedron)


def _fields(m=(0.6, 0.8, 0.0), Ms=8.0e5, extent=1.0, n=3):
    # master built one module-level mesh/m/Ms triple (df.UnitCubeMesh(2,2,2)
    # + a shared df.Function(S3)/Field(S1)) and reused it, including across
    # tests that mutate it via update(); this factory gives every test (old
    # and new) a fresh mesh/fields instead, avoiding that shared-mutable-
    # state hazard under DOLFINx.
    domain = _box(extent, n)
    S3 = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    S1 = fem.functionspace(domain, ("Lagrange", 1))
    m_field = Field(S3, m, name="m")
    Ms_field = Field(S1, Ms, name="Ms")
    return domain, m_field, Ms_field


def _diff(interaction, expected_field):
    """Max deviation between an interaction's average field and expected.

    master: ``H_ext.compute_field().reshape((3, -1)).mean(1)`` unchanged.
    """
    # Task 31: compute_field() is component-blocked; per-component nodal mean
    # is reshape((3, -1)).mean(1), not the old raw-interleaved reshape((-1, 3)).
    H = interaction.compute_field().reshape((3, -1)).mean(1)
    return np.max(np.abs(H - np.asarray(expected_field)))


TOL = 1e-14  # master 1e-14; measured DOLFINx max deviation ~8.9e-16 across
             # every TOL-gated assertion below -> passes verbatim, unloosened.


# ==========================================================================
# MINIMAL-DIFF transcription of master zeeman_test.py (git b5015c5a) --
# the 9 TimeZeeman/DiscreteTimeZeeman/OscillatingZeeman/DipolarField
# functions this file covers, in master's order. dolfin->dolfinx changes are
# annotated inline. See the accounting note in the module docstring above
# for the other 5 master functions.
# ==========================================================================

def test_interaction_accepts_name():
    """
    Check that the interaction accepts a 'name' argument and has a 'name' attribute.
    """
    # df.Expression(("0", "t", "0"), t=0, degree=1) -> plain Python callable
    # (ported TimeZeeman input contract; no legacy mutable-.t Expression).
    def field_function(t):
        return (0.0, t, 0.0)

    zeeman = Zeeman([0, 0, 1], name="MyZeeman")
    assert hasattr(zeeman, "name")
    zeeman = TimeZeeman(field_function, name="MyTimeZeeman")
    assert hasattr(zeeman, "name")
    zeeman = DiscreteTimeZeeman(
        field_function, dt_update=2, name="MyDiscreteTimeZeeman")
    assert hasattr(zeeman, "name")


def test_time_zeeman_init():
    # df.Expression(("0", "t", "0"), t=0, degree=1) -> plain Python callable.
    def field_function(t):
        return (0.0, t, 0.0)

    field_lst = [1, 0, 0]
    field_tpl = (1, 0, 0)
    field_arr = np.array([1, 0, 0])
    # These should work
    TimeZeeman(field_function)
    TimeZeeman(field_function, t_off=1e-9)

    # These should *not* work, since there is no time update
    with pytest.raises(ValueError):
        TimeZeeman(field_lst, t_off=None)
    with pytest.raises(ValueError):
        TimeZeeman(field_tpl, t_off=None)
    with pytest.raises(ValueError):
        TimeZeeman(field_arr, t_off=None)

    # These *should* work, since there is a time update (the field is
    # switched off at some point)
    TimeZeeman(field_lst, t_off=1e-9)
    TimeZeeman(field_tpl, t_off=1e-9)
    TimeZeeman(field_arr, t_off=1e-9)


def test_time_dependent_field_update():
    _, m, Ms = _fields()

    def field_function(t):  # df.Expression(("0", "t", "0"), t=0, degree=1)
        return (0.0, t, 0.0)

    H_ext = TimeZeeman(field_function)
    H_ext.setup(m, Ms, unit_length=1.0)  # df.FunctionSpace(mesh,'DG',0) Ms -> Field(S1)

    assert _diff(H_ext, np.array([0, 0, 0])) < TOL
    H_ext.update(1)
    assert _diff(H_ext, np.array([0, 1, 0])) < TOL


def test_time_dependent_field_switched_off():
    _, m, Ms = _fields()

    # Check the time update (including switching off) with a varying field
    def field_function(t):  # df.Expression(("0", "t", "0"), t=0, degree=1)
        return (0.0, t, 0.0)

    H_ext = TimeZeeman(field_function, t_off=1)
    H_ext.setup(m, Ms, unit_length=1.0)
    assert _diff(H_ext, np.array([0, 0, 0])) < TOL
    assert H_ext.switched_off is False
    H_ext.update(0.9)
    assert _diff(H_ext, np.array([0, 0.9, 0])) < TOL
    assert H_ext.switched_off is False
    H_ext.update(2)
    assert _diff(H_ext, np.array([0, 0, 0])) < TOL  # It's off!
    assert H_ext.switched_off is True

    # The same again with a constant field
    _, m, Ms = _fields()
    a = [42, 0, 5]
    H_ext = TimeZeeman(a, t_off=1)
    H_ext.setup(m, Ms, unit_length=1.0)
    assert _diff(H_ext, a) < TOL
    assert H_ext.switched_off is False
    H_ext.update(0.9)
    assert _diff(H_ext, a) < TOL
    assert H_ext.switched_off is False
    H_ext.update(2)
    assert _diff(H_ext, np.array([0, 0, 0])) < TOL  # It's off!
    assert H_ext.switched_off is True


def test_discrete_time_zeeman_updates_in_intervals():
    _, m, Ms = _fields()

    def field_function(t):  # df.Expression(("0", "t", "0"), t=0, degree=1)
        return (0.0, t, 0.0)

    H_ext = DiscreteTimeZeeman(field_function, dt_update=2)
    H_ext.setup(m, Ms, unit_length=1.0)
    assert _diff(H_ext, np.array([0, 0, 0])) < TOL
    H_ext.update(1)
    assert _diff(H_ext, np.array([0, 0, 0])) < TOL  # not yet updating
    H_ext.update(3)
    assert _diff(H_ext, np.array([0, 3, 0])) < TOL


def test_discrete_time_zeeman_check_arguments_are_sane():
    """
    At least one of the arguments 'dt_update' and 't_off' must be given.
    """
    def field_function(t):  # df.Expression(("1", "2", "3"), degree=1)
        return (1.0, 2.0, 3.0)

    with pytest.raises(ValueError):
        DiscreteTimeZeeman(field_function, dt_update=None, t_off=None)


def test_discrete_time_zeeman_switchoff_only():
    """
    Check that switching off a field works even if no dt_update is
    given (i.e. the field is just a pulse that is switched off after a
    while).
    """
    _, m, Ms = _fields()

    def field_function(t):  # df.Expression(("1", "2", "3"), degree=1)
        return (1.0, 2.0, 3.0)

    H_ext = DiscreteTimeZeeman(field_function, dt_update=None, t_off=2)
    H_ext.setup(m, Ms, unit_length=1.0)
    assert _diff(H_ext, np.array([1, 2, 3])) < TOL
    assert H_ext.switched_off is False
    H_ext.update(1)
    assert _diff(H_ext, np.array([1, 2, 3])) < TOL  # not yet updating
    assert H_ext.switched_off is False
    H_ext.update(2.1)
    assert _diff(H_ext, np.array([0, 0, 0])) < TOL
    assert H_ext.switched_off is True


def test_oscillating_zeeman():
    """
    """
    _, m, Ms = _fields()

    def check_field_at_time(t, val):
        H_osc.update(t)
        # H_osc.compute_field().reshape(3, -1).T: dolfinx compute_field()
        # returns a flat component-blocked array (Task 31); the same
        # reshape/transpose recovers per-node rows as it did under dolfin.
        a = H_osc.compute_field().reshape(3, -1).T
        assert np.allclose(a, val, atol=0, rtol=1e-8)

    H = np.array([1e6, 0.0, 0.0])
    freq = 2e9
    t_off = 10e-9

    H_osc = OscillatingZeeman(H0=H, freq=freq, phase=0, t_off=t_off)
    H_osc.setup(m, Ms, unit_length=1.0)

    # Check that the field has the original value at the end of the
    # first few cycles.
    for i in range(19):  # py2 xrange -> py3 range
        check_field_at_time(i * 1.0 / freq, H)

    # Check that the field is switched off at the specified time (and
    # stays switched off thereafter)
    assert H_osc.switched_off is False
    check_field_at_time(t_off, [0, 0, 0])
    assert H_osc.switched_off is True
    check_field_at_time(t_off + 1e-11, [0, 0, 0])
    assert H_osc.switched_off is True
    check_field_at_time(t_off + 1, [0, 0, 0])
    assert H_osc.switched_off is True

    # Check that the field values vary sinusoidally as expected
    phase = 0.1e-9
    H_osc = OscillatingZeeman(H0=H, freq=freq, phase=phase, t_off=None)
    H_osc.setup(m, Ms, unit_length=1.0)
    for t in np.linspace(0, 20e-9, 100):
        check_field_at_time(t, H * np.cos(2 * np.pi * freq * t + phase))  # math.cos/pi -> np.cos/np.pi


def test_dipolar_field_class(tmp_path, monkeypatch):
    # os.chdir(str(tmpdir)) -> monkeypatch.chdir(tmp_path) (modern pytest
    # idiom for the same "run in a scratch cwd" precaution).
    monkeypatch.chdir(tmp_path)
    H_dipole = DipolarField(pos=[0, 0, 0], m=[1, 0, 0], magnitude=3e9)
    # df.BoxMesh(Point(-50,-50,-50), Point(50,50,50), 20, 20, 20) + a
    # df.VectorFunctionSpace(mesh,'CG',1,dim=3)/df.Function+assign(Constant)
    # m-field and a 'DG',0 Ms field -> the shared _fields() factory (module-
    # wide convention: Ms lives on a Lagrange-1 space throughout this port,
    # see also _oracle_fields()'s "DG_or_S1" comment below).
    _, m_field, Ms_field = _fields(m=(1.0, 0.0, 0.0), Ms=8.6e5, extent=100.0, n=20)
    H_dipole.setup(m_field, Ms_field, unit_length=1e-9)


# ==========================================================================
# ===== NEW under DOLFINx (no master ancestor) ============================
# ==========================================================================
# None of these use a legacy dolfin Expression-with-mutable-.t-attribute;
# see the module docstring above for what each group validates.

# Oracle fixtures live under ``src/finmag/tests/fixtures/``; this file now sits
# at its master path ``src/finmag/energies/zeeman_test.py`` (formerly
# ``src/finmag/tests/test_timezeeman_dolfinx.py``), one directory up and
# across, hence the ``os.pardir`` hop.
_FIXTURE = os.path.join(os.path.dirname(__file__), os.pardir, "tests",
                        "fixtures", "timezeeman_oracle.json")
ORACLE = json.load(open(_FIXTURE))
CASES = ORACLE["cases"]


# --------------------------------------------------------------------------
# TimeZeeman continuous-path / quirk guards
# --------------------------------------------------------------------------

def _analytic_zeeman_energy(Hy, Ms=8.0e5, m_y=0.8, extent=1.0, unit_length=1.0):
    """Analytic uniform-box Zeeman energy for m=(0.6, 0.8, 0), H=(0, Hy, 0):

        E = -mu0 * Ms * (m.H) * V * unit_length**dim
    """
    return -mu0 * Ms * (m_y * Hy) * extent ** 3 * unit_length ** 3


def test_time_dependent_field_update_energy_tracks_analytic():
    """Continuous-path guard (D3): the ``TimeZeeman`` path (which already goes
    through ``set_value``) must keep reporting the analytic energy of its
    current field. Pins the continuous path so the D3 fix to the discrete path
    cannot silently disturb it."""
    _, m, Ms = _fields()

    def field_function(t):
        return (0.0, 1.0e5 + 1.0e5 * t / 1.0e-9, 0.0)

    H_ext = TimeZeeman(field_function)
    H_ext.setup(m, Ms, unit_length=1e-9)

    for t, Hy in ((0.0, 1.0e5), (2e-10, 1.2e5), (4.5e-10, 1.45e5),
                  (1e-9, 2.0e5)):
        if t != 0.0:
            H_ext.update(t)
        assert _diff(H_ext, [0.0, Hy, 0.0]) < 1e-6
        np.testing.assert_allclose(
            H_ext.compute_energy(),
            _analytic_zeeman_energy(Hy, unit_length=1e-9),
            rtol=1e-12, atol=0.0)


def test_time_zeeman_t_off_zero_is_falsy_disabled_quirk():
    """Preserved legacy quirk: ``if self.t_off and t >= self.t_off`` never
    triggers for ``t_off=0.0`` (falsy), so the field is never switched off."""
    _, m, Ms = _fields()

    def field_function(t):
        return (0.0, t, 0.0)

    H_ext = TimeZeeman(field_function, t_off=0.0)
    H_ext.setup(m, Ms, unit_length=1.0)
    H_ext.update(5.0)
    assert H_ext.switched_off is False
    assert _diff(H_ext, [0, 5.0, 0]) < TOL


def test_discrete_time_zeeman_continuous_after_first_crossing_quirk():
    """Preserved legacy quirk: ``t_last_update`` never advances, so once ``t``
    first reaches ``dt_update`` every later ``update(t)`` call refreshes
    the field again (not just once every ``dt_update``)."""
    _, m, Ms = _fields()

    def field_function(t):
        return (0.0, t, 0.0)

    H_ext = DiscreteTimeZeeman(field_function, dt_update=2)
    H_ext.setup(m, Ms, unit_length=1.0)
    for t in (1.0, 2.0, 2.5, 4.0, 4.1, 4.5):
        H_ext.update(t)
        expected = 0.0 if t < 2.0 else t
        assert _diff(H_ext, [0, expected, 0]) < TOL, (t, H_ext.compute_field())


# --------------------------------------------------------------------------
# DiscreteTimeZeeman stale-energy DIVERGENCE PIN and corrected-behaviour tests
# (owner-approved fix, acceptance register D3, approved 2026-07-23)
# --------------------------------------------------------------------------

# The setup-time (t=0) energy of the D3 fixture below: a 1x1x1 box with
# unit_length=1e-9, m=(0.6, 0.8, 0), Ms=8e5 A/m and H(0)=(0, 1e5, 0), i.e.
# E = -mu0 * 8e5 * (0.8 * 1e5) * 1**3 * (1e-9)**3.  Legacy froze
# compute_energy() at exactly this number for all later t.
_D3_LEGACY_STALE_ENERGY = -8.042477193189932e-23


def _d3_discrete_zeeman(unit_length=1e-9, dt_update=2e-10):
    """The D3 fixture: a deliberately NONZERO setup field, so the magnitude of
    the legacy staleness is visible (the committed oracle fixture's field is
    zero at t=0, which hides it -- its correct energies are ~1e-37 J against
    an absolute tolerance of 1e-18, so it does not discriminate the defect)."""
    _, m, Ms = _fields()

    def field_function(t):
        return (0.0, 1.0e5 + 1.0e5 * t / 1.0e-9, 0.0)

    H_ext = DiscreteTimeZeeman(field_function, dt_update=dt_update)
    H_ext.setup(m, Ms, unit_length=unit_length)
    return H_ext


def test_discrete_time_zeeman_energy_tracks_field_after_interval_update():
    """D3 fix: after an interval update ``compute_energy()`` reports the energy
    of the field that ``compute_field()`` reports, i.e. the analytic
    ``E = -mu0 * Ms * (m.H) * V * unit_length**dim`` -- not the setup-time
    value.  Asserted against the closed form, not merely "it changed"."""
    H_ext = _d3_discrete_zeeman()

    for t, Hy in ((0.0, 1.0e5), (1e-10, 1.0e5), (2e-10, 1.2e5),
                  (2.5e-10, 1.25e5), (4e-10, 1.4e5), (4.1e-10, 1.41e5),
                  (4.5e-10, 1.45e5), (1e-9, 2.0e5)):
        if t != 0.0:
            H_ext.update(t)
        assert _diff(H_ext, [0.0, Hy, 0.0]) < 1e-6, (t, H_ext.compute_field())
        np.testing.assert_allclose(
            H_ext.compute_energy(),
            _analytic_zeeman_energy(Hy, unit_length=1e-9),
            rtol=1e-12, atol=0.0)


def test_discrete_time_zeeman_energy_diverges_from_legacy_stale_value():
    """DIVERGENCE PIN (OWNER-APPROVED FIX, acceptance register D3, approved
    2026-07-25): legacy's ``DiscreteTimeZeeman.update`` rebound ``self.H`` to a
    brand-new ``Field`` object -- frozen oracle ``ba928093``:

        ``self.H = Field(dg_vector_functionspace, self.value, name='H_ext')``

    -- instead of calling ``set_value()``.  The cached UFL energy form
    ``self.E = -mu0 * Ms.f * dot(m.f, H.f)`` is built once in ``setup()``
    against the *original* ``H`` ``Function`` object, so after the rebind it
    kept assembling the discarded setup-time field forever: legacy returned
    exactly ``E(H(t=0))`` at every later ``t``, while ``compute_field()`` and
    ``energy_density()`` (which read ``self.H``'s array by attribute lookup at
    call time) correctly showed ``H(t)``.  The relative error was 16.7% at the
    first interval crossing here, 50% at t=1ns, and is unbounded in general
    (100% when ``H(0)=0``, and the wrong *sign* when ``m.H(t)`` flips).

    Under D3 the port now returns the energy of the current field.  This test
    records the resulting divergence from legacy so the behaviour change is
    documented rather than silently erased.  Quirk 1 (``t_last_update`` never
    advancing) is a separate, still-preserved legacy behaviour and is pinned by
    ``test_discrete_time_zeeman_continuous_after_first_crossing_quirk``.
    """
    H_ext = _d3_discrete_zeeman()

    # setup-time energy: this is the value legacy froze at forever
    E_legacy = H_ext.compute_energy()
    np.testing.assert_allclose(E_legacy, _D3_LEGACY_STALE_ENERGY, rtol=1e-9)

    H_ext.update(1e-9)

    # the field still agrees with legacy -- the legacy defect never touched it
    assert _diff(H_ext, [0.0, 2.0e5, 0.0]) < 1e-6

    E_now = H_ext.compute_energy()

    # the divergence from the legacy stale value is large, not noise
    assert abs(E_now - E_legacy) > 0.4 * abs(E_legacy), (E_now, E_legacy)

    # structural identity: the energy is linear in H, so for this
    # linear-in-t field the ratio is exactly the field ratio H_y(t)/H_y(0) = 2
    np.testing.assert_allclose(E_now / E_legacy, 2.0, rtol=1e-9)

    # ... and it equals a freshly built static Zeeman on the very same field
    _, m2, Ms2 = _fields()
    ref = Zeeman((0.0, 2.0e5, 0.0))
    ref.setup(m2, Ms2, unit_length=1e-9)
    np.testing.assert_allclose(E_now, ref.compute_energy(), rtol=1e-12)


def test_discrete_time_zeeman_d3_fix_changes_no_field_value():
    """Field-invariance guard (D3): the fix only re-forms the cached energy
    form; every field surface (``compute_field()``, ``average_field()``) must
    stay BIT-IDENTICAL to the pre-fix brand-new-``Field`` assignment path, for
    BOTH supported ``field_function`` contracts (constant vector and
    spatially-varying callable).  This is the guard that protects dynamics --
    ``EffectiveField``/LLG only ever see ``compute_field()``.

    The pre-fix path was ``self.H = Field(functionspace, value)``; the fix
    routes through ``set_value`` -> ``self.H.set(value)`` (in place).  Both are
    bit-identical to a freshly constructed static ``Zeeman(value)`` on the same
    field (measured max|dH| = 0.0 at every step), so that fresh build is the
    invariant reference here.  ``assert_array_equal`` (exact equality) is
    deliberate: a no-op must be bit-exact, not merely close.
    """
    # (a) constant-vector contract
    _, m, Ms = _fields()

    def constant_vector(t):
        return (0.0, t, 0.0)

    H_ext = DiscreteTimeZeeman(constant_vector, dt_update=2)
    H_ext.setup(m, Ms, unit_length=1.0)
    for t in (1.0, 2.0, 2.5, 4.0, 4.1, 4.5):
        H_ext.update(t)
        held = 0.0 if t < 2.0 else t  # quirk-1 continuous-refresh value
        _, mr, Msr = _fields()
        ref = Zeeman((0.0, held, 0.0))
        ref.setup(mr, Msr, unit_length=1.0)
        np.testing.assert_array_equal(
            H_ext.compute_field(), ref.compute_field())
        np.testing.assert_array_equal(
            H_ext.average_field(), ref.average_field())

    # (b) spatially-varying callable contract
    _, m2, Ms2 = _fields()

    def spatial_callable(t):
        def field(x):
            return np.vstack((
                np.zeros(x.shape[1]),
                1.0e5 * (1.0 - 2.0 * x[0]) * (1.0 + t / 1.0e-9),
                np.zeros(x.shape[1])))
        return field

    H_sp = DiscreteTimeZeeman(spatial_callable, dt_update=2e-10)
    H_sp.setup(m2, Ms2, unit_length=1e-9)
    for t in (2e-10, 4.5e-10):
        H_sp.update(t)
        _, mr, Msr = _fields()
        ref = Zeeman(spatial_callable(t))
        ref.setup(mr, Msr, unit_length=1e-9)
        np.testing.assert_array_equal(
            H_sp.compute_field(), ref.compute_field())
        np.testing.assert_array_equal(
            H_sp.average_field(), ref.average_field())


def test_discrete_time_zeeman_energy_is_zero_after_switch_off():
    """Switch-off guard (D3): the ``t_off`` branch goes through
    ``switch_off()`` -> ``set_value((0,0,0))``, which re-forms ``self.E``, so
    the energy of the zeroed field is exactly 0.0."""
    _, m, Ms = _fields()

    def field_function(t):
        return (0.0, 1.0e5 + 1.0e5 * t / 1.0e-9, 0.0)

    H_ext = DiscreteTimeZeeman(field_function, dt_update=2e-10, t_off=6e-10)
    H_ext.setup(m, Ms, unit_length=1e-9)
    np.testing.assert_allclose(
        H_ext.compute_energy(), _D3_LEGACY_STALE_ENERGY, rtol=1e-9)

    H_ext.update(1e-9)
    assert H_ext.switched_off is True
    assert _diff(H_ext, [0.0, 0.0, 0.0]) < TOL
    assert H_ext.compute_energy() == 0.0


# --------------------------------------------------------------------------
# TimeZeemanPython (no master ancestor: master's zeeman_test.py never
# imports/exercises this class)
# --------------------------------------------------------------------------

def test_time_zeeman_python_scales_cached_spatial_pattern():
    _, m, Ms = _fields()

    calls = []

    def time_fun(t):
        calls.append(t)
        return 2.0 * t

    H0 = (1.0, -1.0, 0.5)
    tzp = TimeZeemanPython(H0, time_fun)
    tzp.setup(m, Ms, unit_length=1.0)
    assert _diff(tzp, [0.0, 0.0, 0.0]) < TOL  # time_fun(0.0) == 0.0

    tzp.update(3.0)
    assert _diff(tzp, np.array(H0) * 6.0) < TOL

    # compute_energy/energy_density are inherited unchanged from Zeeman and
    # must reflect the current (rescaled) field -- as, since the D3 fix, does
    # DiscreteTimeZeeman's (see the D3 divergence pin above).
    E = tzp.compute_energy()
    assert np.isfinite(E) and E != 0.0

    tzp.update(0.5)
    assert _diff(tzp, np.array(H0) * 1.0) < TOL


def test_time_zeeman_python_vector_time_fun_is_deferred_by_name():
    """Pin (whole-branch review finding 1): the legacy scalar-spatial-
    envelope-with-vector-``time_fun`` branch is not ported. ``setup`` must
    raise a named ``NotImplementedError`` upfront, not a generic
    ``ValueError``/``TypeError`` discovered mid-integration."""
    _, m, Ms = _fields()

    def vector_time_fun(t):
        return (1.0, 0.0, 0.0)

    tzp = TimeZeemanPython((1.0, 0.0, 0.0), vector_time_fun)
    with pytest.raises(NotImplementedError, match="vector-valued time_fun"):
        tzp.setup(m, Ms, unit_length=1.0)

    # the scalar envelope path is unaffected by the new gate.
    tzp_scalar = TimeZeemanPython((1.0, 0.0, 0.0), lambda t: 2.0 * t)
    tzp_scalar.setup(m, Ms, unit_length=1.0)
    tzp_scalar.update(3.0)
    assert _diff(tzp_scalar, [6.0, 0.0, 0.0]) < TOL


def test_time_zeeman_python_switch_off():
    _, m, Ms = _fields()
    tzp = TimeZeemanPython((1.0, 0.0, 0.0), lambda t: 1.0, t_off=1.0)
    tzp.setup(m, Ms, unit_length=1.0)
    tzp.update(0.5)
    assert _diff(tzp, [1.0, 0.0, 0.0]) < TOL
    tzp.update(2.0)
    assert tzp.switched_off is True
    assert _diff(tzp, [0.0, 0.0, 0.0]) < TOL


# --------------------------------------------------------------------------
# DipolarField -- stronger analytic tests superseding master's assertion-free
# smoke test (transcribed verbatim above as test_dipolar_field_class)
# --------------------------------------------------------------------------

def test_dipolar_field_matches_closed_form():
    domain, m, Ms = _fields(extent=100.0, n=8)
    H_dipole = DipolarField(pos=[0, 0, 0], m=[1, 0, 0], magnitude=3e9)
    H_dipole.setup(m, Ms, unit_length=1e-9)

    # Task 31: component-blocked field -> owned-vertex per-node rows, paired
    # with the matching owned-vertex coordinates.
    n_owned = domain.geometry.index_map().size_local
    coords = domain.geometry.x[:n_owned, :3]
    H = H_dipole.compute_field().reshape((3, -1)).T

    moment = np.array([3e9, 0.0, 0.0])
    # Skip points too close to the origin dipole (closed form singular there).
    r = np.linalg.norm(coords, axis=1)
    mask = r > 5.0
    v = -coords[mask]
    rr = r[mask]
    dotted = v @ moment
    expected = (1.0 / (4 * np.pi)) * (
        3 * v * dotted[:, None] / rr[:, None] ** 5 - moment / rr[:, None] ** 3)
    np.testing.assert_allclose(H[mask], expected, atol=0, rtol=1e-8)


def test_dipolar_field_direction_only_with_magnitude():
    _, m, Ms = _fields()
    H_dipole = DipolarField(pos=[10, 10, 10], m=[2, 0, 0], magnitude=5.0)
    np.testing.assert_allclose(H_dipole.m, [5.0, 0.0, 0.0])


# --------------------------------------------------------------------------
# oracle fixtures
# --------------------------------------------------------------------------

def _oracle_box_mesh():
    return mesh.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
        [3, 3, 3], mesh.CellType.tetrahedron)


def _oracle_fields():
    domain = _oracle_box_mesh()
    S3 = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    DG_or_S1 = fem.functionspace(domain, ("Lagrange", 1))
    m = Field(S3, (0.6, 0.8, 0.0), name="m")
    Ms = Field(DG_or_S1, 8.0e5, name="Ms")
    return domain, m, Ms


def test_oracle_time_zeeman_sequence_matches_legacy():
    case = CASES["time_zeeman"]
    domain, m, Ms = _oracle_fields()
    A = 2.0e5
    w = 2.0 * np.pi * 1.0e9

    def field_function(t):
        def spatial(x):
            zeros = np.zeros(x.shape[1])
            return np.vstack((zeros, A * x[0] * np.cos(w * t), zeros))
        return spatial

    tz = TimeZeeman(field_function)
    tz.setup(m, Ms, unit_length=1e-9)

    # Task 31: sort owned-vertex coordinates; blocked field rows follow suit.
    n_owned = domain.geometry.index_map().size_local
    coords = domain.geometry.x[:n_owned, :3]
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))

    ref_coords = np.asarray(case["coordinates"]["values"])
    np.testing.assert_allclose(coords[order], ref_coords, rtol=0, atol=1e-9)

    for t, snap in zip(case["t_values"], case["snapshots"]):
        if t != 0.0:
            tz.update(t)
        E = tz.compute_energy()
        H = tz.compute_field().reshape((3, -1)).T[order]

        e_q = snap["scalar_quantities"][0]
        h_q = snap["quantities"][0]
        np.testing.assert_allclose(
            E, e_q["value"], atol=e_q["tolerances"]["absolute"],
            rtol=e_q["tolerances"]["relative"])
        np.testing.assert_allclose(
            H, np.asarray(h_q["values"]), atol=h_q["tolerances"]["absolute"],
            rtol=h_q["tolerances"]["relative"])


def test_oracle_discrete_time_zeeman_sequence_matches_legacy():
    case = CASES["discrete_time_zeeman"]
    domain, m, Ms = _oracle_fields()

    def field_function(t):
        return (0.0, t, 0.0)

    dtz = DiscreteTimeZeeman(field_function, dt_update=2e-10)
    dtz.setup(m, Ms, unit_length=1e-9)

    # Task 31: sort owned-vertex coordinates; blocked field rows follow suit.
    n_owned = domain.geometry.index_map().size_local
    coords = domain.geometry.x[:n_owned, :3]
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))

    for t, snap in zip(case["t_values"], case["snapshots"]):
        if t != 0.0:
            dtz.update(t)
        E = dtz.compute_energy()
        H = dtz.compute_field().reshape((3, -1)).T[order]

        e_q = snap["scalar_quantities"][0]
        h_q = snap["quantities"][0]
        np.testing.assert_allclose(
            E, e_q["value"], atol=e_q["tolerances"]["absolute"],
            rtol=e_q["tolerances"]["relative"])
        np.testing.assert_allclose(
            H, np.asarray(h_q["values"]), atol=h_q["tolerances"]["absolute"],
            rtol=h_q["tolerances"]["relative"])


# --------------------------------------------------------------------------
# EffectiveField auto-connection with the REAL TimeZeeman family (Task 6
# preserved contract, previously only exercised with a FakeTimeZeeman
# double -- see tests/test_effective_field.py, formerly
# test_effective_field_dolfinx.py)
# --------------------------------------------------------------------------

def test_real_timezeeman_auto_connects_without_explicit_with_time_update():
    domain = mesh.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (5.0, 5.0, 5.0)], [2, 2, 2],
        mesh.CellType.tetrahedron)
    sim = Simulation(domain, 8.6e5, unit_length=1e-9, name="tz_autoconnect")
    sim.set_m((1.0, 0.0, 0.0))
    sim.alpha = 0.5

    osc = OscillatingZeeman(H0=(0.0, 0.0, 1e5), freq=1e8, phase=0.0, t_off=None)
    sim.add(osc)  # no explicit with_time_update

    # compute() without t must raise (Task 6 behavior) since a real
    # TimeZeeman subclass is present and needs a time update.
    with pytest.raises(ValueError):
        sim.llg.effective_field.compute()

    # With t given, it must not raise, and the field must equal H0 (t=0):
    # OscillatingZeeman(phase=0.0) gives cos(2*pi*freq*0 + 0) == 1, and it is
    # the only interaction added, so H_eff == H0 everywhere.
    H_at_0 = sim.llg.effective_field.compute(t=0.0)
    assert np.isfinite(H_at_0).all()
    np.testing.assert_allclose(
        H_at_0.reshape((3, -1)).T, np.broadcast_to((0.0, 0.0, 1e5), (H_at_0.size // 3, 3)),
        rtol=1e-12, atol=0.0)


def test_real_timezeeman_field_changes_during_run_until():
    domain = mesh.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (5.0, 5.0, 5.0)], [2, 2, 2],
        mesh.CellType.tetrahedron)
    sim = Simulation(domain, 8.6e5, unit_length=1e-9, name="tz_run_until")
    sim.set_m((1.0, 0.0, 0.0))
    sim.alpha = 0.5

    freq = 1e8
    osc = OscillatingZeeman(H0=(0.0, 0.0, 1e5), freq=freq, phase=0.0, t_off=None)
    sim.add(osc)

    H_before = osc.compute_field().copy()
    t_end = 1e-9
    sim.run_until(t_end)
    H_after = osc.compute_field().copy()

    expected_scale = np.cos(2.0 * np.pi * freq * sim.t)
    expected = np.array([0.0, 0.0, 1e5]) * expected_scale
    assert not np.allclose(H_before, H_after)
    np.testing.assert_allclose(
        H_after.reshape((3, -1)).T, np.broadcast_to(expected, (H_after.size // 3, 3)),
        atol=1.0, rtol=1e-6)


# ===== NOT PORTED (carried verbatim from master b5015c5a; expected to fail) =====
#
# Reason: ``test_compare_stray_field_of_sphere_with_dipolar_field`` is the one
# function of master's 14 in ``src/finmag/energies/zeeman_test.py`` that is
# covered NEITHER by the transcription above NOR by the static-Zeeman aggregate
# ``src/finmag/tests/test_energies.py`` (see this module's docstring
# accounting). It depends on ``finmag.example.sphere_inside_airbox`` (deferred
# under DOLFINx: raises ``NotImplementedError`` by name) and on
# ``Simulation.get_field_as_dolfin_function(..., region=...)``. Carried verbatim
# from ``git show b5015c5a:src/finmag/energies/zeeman_test.py`` so the gap stays
# visible in the non-gating inventory lane instead of vanishing with master's
# file. Master's own ``slow``/``xfail(reason='dolfin 1.5')`` markers are kept
# unchanged (xfail pins must not change in this re-layout).

import logging  # master imported logging at module scope, for ``logger`` below
from math import pi  # master imported pi from math at module scope

try:
    # master had this as a plain module-level import. Under DOLFINx it raises
    # NotImplementedError at import time, which would break COLLECTION of this
    # whole file, so it is guarded to None here; the carried test below then
    # fails visibly at call time instead (the intended inventory-lane
    # behaviour). This is the only structural change to master's code.
    from finmag.example import sphere_inside_airbox
except NotImplementedError:  # pragma: no cover - deferred example machinery
    sphere_inside_airbox = None

logger = logging.getLogger('finmag')  # master module-level logger


def compute_field_diffs(sim):
    vals_demag = sim.get_field_as_dolfin_function(
        'Demag', region='air').vector().array().reshape(3, -1)
    vals_dipole = sim.get_field_as_dolfin_function(
        'DipolarField', region='air').vector().array().reshape(3, -1)
    absdiffs = np.linalg.norm(vals_demag - vals_dipole, axis=0)
    reldiffs = absdiffs / np.linalg.norm(vals_dipole, axis=0)
    return absdiffs, reldiffs


@pytest.mark.not_ported
@pytest.mark.slow
@pytest.mark.xfail(reason='dolfin 1.5')
def test_compare_stray_field_of_sphere_with_dipolar_field(tmpdir, debug=False):
    """
    Check that the stray field of a sphere in an 'airbox'
    is close to the field of a point dipole with the same
    magnetic moment.

    """
    os.chdir(str(tmpdir))

    # Create a mesh of a sphere enclosed in an "airbox"
    m_init = [7, -4, 3]  # some random magnetisation direction
    center_sphere = [0, 0, 0]
    r_sphere = 3
    r_shell = 30
    l_box = 100
    maxh_sphere = 2.5
    maxh_shell = None
    maxh_box = 10.0
    Ms_sphere = 8.6e5

    sim = sphere_inside_airbox(
        r_sphere, r_shell, l_box, maxh_sphere, maxh_shell, maxh_box, center_sphere, m_init)
    if debug:
        sim.render_scene(field_name='Demag', region='air',
                         representation='Outline', outfile='ddd_demag_field_air.png')
        sim.render_scene(field_name='Demag', region='sphere',
                         representation='Outline', outfile='ddd_demag_field_sphere.png')

    # Add an external field representing a point dipole
    # (with the same magnetic moment as the sphere).
    dipole_magnitude = Ms_sphere * 4 / 3 * pi * r_sphere ** 3
    logger.debug("dipole_magnitude = {}".format(dipole_magnitude))
    H_dipole = DipolarField(
        pos=[0, 0, 0], m=m_init, magnitude=dipole_magnitude)
    sim.add(H_dipole)

    # Check that the absolute and relative difference between the
    # stray field of the sphere and the field of the point dipole
    # are below a given tolerance.
    absdiffs, reldiffs = compute_field_diffs(sim)
    assert np.max(reldiffs) < 0.4
    assert np.mean(reldiffs) < 0.15
    assert np.max(absdiffs) < 140.0

    print(np.max(reldiffs), np.mean(reldiffs), np.max(absdiffs))  # py2 print stmt -> py3 print()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
