"""Focused production tests for the direct DOLFINx EffectiveField port.

MASTER -> PORT MAPPING (SR1 P5.2 audit restoration)
====================================================

This file aggregates TWO master ancestors, each contributing exactly one test
function. Both are accounted for below; there is no genuine gap.

- ``src/finmag/tests/test_effective_field.py`` (git ``b5015c5a``):
    * ``test_get_interaction_list`` -> ``test_get_interaction_list`` (this
      file, restored by audit). Faithful, near-verbatim transcription: same
      ``finmag.example.barmini()`` / ``Simulation.get_interaction_list`` /
      ``Simulation.remove_interaction`` / ``finmag.sim_with`` API surface
      (identical names, no divergence) and the same assertions. The only
      change is the mesh constructor, ``dolfin.IntervalMesh(10, 0, 1)`` ->
      ``dolfinx.mesh.create_interval(MPI.COMM_WORLD, 10, [0.0, 1.0])``
      (the standard dolfin->dolfinx mesh-API substitution used throughout
      this port; see e.g. ``test_llg_dolfinx.py``). Measured: passes
      unchanged under DOLFINx (barmini's FK demag + Exchange interactions
      give ``get_interaction_list() == ['Demag', 'Exchange']`` sorted, same
      as legacy's ``sorted(...)`` contract).

- ``src/finmag/physics/tests/test_effective_field.py`` (git ``b5015c5a``):
    * ``test_effective_field_compute_returns_copy`` -> covered, ADAPTED, by
      ``test_compute_returns_a_copy_not_a_live_reference`` (this file,
      pre-existing). Master exercises the copy-not-reference contract of
      ``EffectiveField.compute()`` through a full ``barmini()`` Simulation
      plus ``run_until(1e-12)`` integration step; the port exercises the
      identical contract directly against ``EffectiveField`` using a
      ``RecordingInteraction`` double, because this file is a focused
      physics-layer unit-test file for ``EffectiveField`` itself (Simulation
      *integration* -- constructing a full barmini + advancing the SciPy /
      Sundials integrator -- is covered by the Simulation test suite,
      ``test_simulation_dolfinx.py``, not duplicated here). Tolerance is
      UNCHANGED: ``atol=0, rtol=1e-8``, identical to master's assertion.

2 master functions total (1 per file): 1 restored faithfully, 1
covered-elsewhere-adapted with reasoning above. No dropped coverage remains.
"""

import sys

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

import finmag
from finmag.energies import Exchange, TimeZeeman, UniaxialAnisotropy, Zeeman
from finmag.energies.energy_base import mu0
from finmag.field import Field
from finmag.physics.effective_field import EffectiveField
from finmag.physics.errors import UnknownInteraction


def _fields(m=(1.0, 0.0, 0.0), Ms=2.5, cells=2):
    domain = mesh.create_unit_square(MPI.COMM_WORLD, cells, cells)
    vector_space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    scalar_space = fem.functionspace(domain, ("DG", 0))
    return (
        domain,
        Field(vector_space, m, name="m"),
        Field(scalar_space, Ms, name="Ms"),
    )


class RecordingInteraction:
    """A plain-Python interaction double exercising the registry without FEM."""

    def __init__(self, name, field_value=0.0, energy_value=0.0, in_jacobian=False):
        self.name = name
        self.field_value = field_value
        self.energy_value = energy_value
        self.in_jacobian = in_jacobian
        self.setup_calls = []

    def setup(self, m, Ms, unit_length):
        self.setup_calls.append((m, Ms, unit_length))
        self.size = m.as_array().size
        return self

    def compute_field(self):
        return np.full(self.size, self.field_value)

    def compute_energy(self):
        return self.energy_value


def test_ported_effective_field_does_not_load_legacy_dolfin():
    assert EffectiveField.__module__ == "finmag.physics.effective_field"
    assert "dolfin" not in sys.modules


def test_registry_add_get_exists_all_remove_with_plain_doubles():
    _, m, Ms = _fields()
    effective_field = EffectiveField(m, Ms, unit_length=1.0)

    a = RecordingInteraction("A", field_value=1.0, energy_value=2.0)
    b = RecordingInteraction("B", field_value=3.0, energy_value=5.0)
    effective_field.add(a)
    effective_field.add(b)

    assert effective_field.exists("A")
    assert effective_field.exists("B")
    assert not effective_field.exists("C")
    assert effective_field.all() == ["A", "B"]
    assert effective_field.get("A") is a
    assert effective_field.get("B") is b
    assert a.setup_calls == [(m, Ms, 1.0)]

    effective_field.remove("A")
    assert effective_field.all() == ["B"]
    assert not effective_field.exists("A")


def test_add_rejects_duplicate_interaction_names():
    _, m, Ms = _fields()
    effective_field = EffectiveField(m, Ms, unit_length=1.0)
    effective_field.add(RecordingInteraction("A"))

    with pytest.raises(ValueError, match="unique"):
        effective_field.add(RecordingInteraction("A"))


def test_get_and_remove_raise_unknown_interaction_for_missing_name():
    _, m, Ms = _fields()
    effective_field = EffectiveField(m, Ms, unit_length=1.0)

    with pytest.raises(UnknownInteraction):
        effective_field.get("missing")
    with pytest.raises(UnknownInteraction):
        effective_field.remove("missing")


def test_compute_and_total_energy_sum_plain_doubles():
    _, m, Ms = _fields()
    effective_field = EffectiveField(m, Ms, unit_length=1.0)
    effective_field.add(RecordingInteraction("A", field_value=1.0, energy_value=2.0))
    effective_field.add(RecordingInteraction("B", field_value=3.0, energy_value=5.0))

    H = effective_field.compute()
    assert np.allclose(H, 4.0)
    assert effective_field.total_energy() == pytest.approx(7.0)


def test_compute_jacobian_only_includes_only_jacobian_interactions():
    _, m, Ms = _fields()
    effective_field = EffectiveField(m, Ms, unit_length=1.0)
    effective_field.add(
        RecordingInteraction("A", field_value=1.0, in_jacobian=True)
    )
    effective_field.add(
        RecordingInteraction("B", field_value=10.0, in_jacobian=False)
    )

    H = effective_field.compute_jacobian_only(t=None)
    assert np.allclose(H, 1.0)


def test_compute_returns_a_copy_not_a_live_reference():
    """Regression test mirroring the legacy compute()-returns-a-copy check."""
    _, m, Ms = _fields()
    effective_field = EffectiveField(m, Ms, unit_length=1.0)
    interaction = RecordingInteraction("A", field_value=1.0)
    effective_field.add(interaction)

    h0 = effective_field.compute()
    h0_copy = h0.copy()

    interaction.field_value = 42.0
    h1 = effective_field.compute()

    assert np.allclose(h0, h0_copy, atol=0, rtol=1e-8)
    assert not np.allclose(h0, h1, atol=0, rtol=1e-8)
    # Mutating the returned array must not corrupt internal state either.
    h1[:] = -999.0
    h2 = effective_field.compute()
    assert np.allclose(h2, 42.0)


def test_time_update_required_but_no_t_given_raises():
    _, m, Ms = _fields()
    effective_field = EffectiveField(m, Ms, unit_length=1.0)
    interaction = RecordingInteraction("A", field_value=1.0)
    calls = []
    effective_field.add(interaction, with_time_update=lambda t: calls.append(t))

    with pytest.raises(ValueError, match="time step"):
        effective_field.update()
    assert calls == []


def test_time_update_callback_runs_with_given_t():
    _, m, Ms = _fields()
    effective_field = EffectiveField(m, Ms, unit_length=1.0)
    interaction = RecordingInteraction("A", field_value=1.0)
    calls = []
    effective_field.add(interaction, with_time_update=lambda t: calls.append(t))

    effective_field.compute(t=1.5e-9)
    assert calls == [1.5e-9]


def test_no_time_update_needed_allows_omitting_t():
    _, m, Ms = _fields()
    effective_field = EffectiveField(m, Ms, unit_length=1.0)
    effective_field.add(RecordingInteraction("A", field_value=1.0))

    # Must not raise even though t is omitted.
    H = effective_field.compute()
    assert np.allclose(H, 1.0)


def test_timezeeman_subclass_auto_connects_without_explicit_with_time_update():
    """TimeZeeman itself cannot be constructed (deferred), but a subclass that
    overrides __init__ is still `isinstance(..., TimeZeeman)` and exercises the
    automatic with_time_update connection documented for real TimeZeeman use."""

    class FakeTimeZeeman(TimeZeeman):
        def __init__(self, name):
            self.name = name
            self.in_jacobian = False
            self.update_calls = []

        def setup(self, m, Ms, unit_length):
            self.size = m.as_array().size
            return self

        def compute_field(self):
            return np.full(self.size, 1.0)

        def compute_energy(self):
            return 0.0

        def update(self, t):
            self.update_calls.append(t)

    _, m, Ms = _fields()
    effective_field = EffectiveField(m, Ms, unit_length=1.0)
    interaction = FakeTimeZeeman("H_time")
    effective_field.add(interaction)

    effective_field.compute(t=3.0)
    assert interaction.update_calls == [3.0]


def test_total_field_and_energy_with_ported_interactions():
    _, m, Ms = _fields(m=(0.6, 0.0, 0.8), Ms=8.0e5, cells=2)
    unit_length = 1.0e-9
    effective_field = EffectiveField(m, Ms, unit_length)

    H_applied = np.array((1.0e5, -2.0e4, 3.0e4))
    zeeman = Zeeman(H_applied, name="Zeeman")
    exchange = Exchange(13.0e-12, name="Exchange")
    anisotropy = UniaxialAnisotropy(4.0e3, (0.0, 0.0, 1.0), K2=0.0, name="Anisotropy")

    effective_field.add(zeeman)
    effective_field.add(exchange)
    effective_field.add(anisotropy)

    assert effective_field.all() == ["Anisotropy", "Exchange", "Zeeman"]

    H_total = effective_field.compute()
    expected_total = (
        zeeman.compute_field() + exchange.compute_field() + anisotropy.compute_field()
    )
    assert np.allclose(H_total, expected_total)

    expected_energy = (
        zeeman.compute_energy() + exchange.compute_energy() + anisotropy.compute_energy()
    )
    assert effective_field.total_energy() == pytest.approx(expected_energy)

    jacobian_only = effective_field.compute_jacobian_only(t=None)
    expected_jacobian = exchange.compute_field() + anisotropy.compute_field()
    assert np.allclose(jacobian_only, expected_jacobian)
    assert not np.allclose(jacobian_only, H_total)


def test_removing_an_interaction_changes_effective_field_not_only_energy():
    _, m, Ms = _fields(m=(0.6, 0.0, 0.8), Ms=8.0e5, cells=2)
    # A spatially varying m gives Exchange a non-zero field contribution, so
    # removing it must change H_eff itself, not merely the reported energy.
    m.set(
        lambda x: np.vstack(
            (0.6 + 0.1 * x[0], np.zeros(x.shape[1]), 0.8 * np.ones(x.shape[1]))
        )
    )
    effective_field = EffectiveField(m, Ms, unit_length=1.0e-9)

    zeeman = Zeeman((1.0e5, -2.0e4, 3.0e4), name="Zeeman")
    exchange = Exchange(13.0e-12, name="Exchange")
    effective_field.add(zeeman)
    effective_field.add(exchange)

    H_before = effective_field.compute()
    energy_before = effective_field.total_energy()
    exchange_energy = exchange.compute_energy()
    assert abs(exchange_energy) > 1e-25  # sanity: the perturbation is not inert

    effective_field.remove("Exchange")
    H_after = effective_field.compute()
    energy_after = effective_field.total_energy()

    assert not np.allclose(H_before, H_after)
    assert energy_before - energy_after == pytest.approx(exchange_energy)
    # And the surviving field values are exactly the Zeeman-only contribution.
    assert np.allclose(H_after, zeeman.compute_field())


def test_get_dolfin_function_returns_a_function_on_ms_space():
    _, m, Ms = _fields(m=(1.0, 0.0, 0.0), Ms=2.5, cells=2)
    effective_field = EffectiveField(m, Ms, unit_length=1.0)
    H_applied = (1.0, -2.0, 3.0)
    zeeman = Zeeman(H_applied, name="Zeeman")
    effective_field.add(zeeman)

    result = effective_field.get_dolfin_function("Zeeman")
    assert result.function_space == m.functionspace
    wrapped = Field(m.functionspace, result)
    assert np.allclose(wrapped.as_array().reshape((-1, 3)), H_applied)


def test_get_dolfin_function_rejects_region_argument_explicitly():
    _, m, Ms = _fields()
    effective_field = EffectiveField(m, Ms, unit_length=1.0)
    effective_field.add(Zeeman((1.0, 0.0, 0.0)))

    with pytest.raises(NotImplementedError, match="region"):
        effective_field.get_dolfin_function("Zeeman", region="core")


# ===== Restored from master (SR1 P5.2 audit) =====
#
# ``src/finmag/tests/test_effective_field.py`` (git b5015c5a),
# ``test_get_interaction_list``. This exercises the Simulation-level registry
# API (``get_interaction_list`` / ``remove_interaction`` / ``sim_with``),
# which is a thin pass-through to the ``EffectiveField`` registry tested
# directly above, but the audit flagged that the port silently dropped this
# Simulation-facing coverage. Restored here near-verbatim; the only change is
# the mesh constructor (see module docstring mapping table).
def test_get_interaction_list():
    # has bar mini example Demag and Exchange?
    s = finmag.example.barmini()
    lst = s.get_interaction_list()
    assert 'Exchange' in lst
    assert 'Demag' in lst
    assert len(lst) == 2

    # Let's remove one and check again
    s.remove_interaction('Exchange')
    assert s.get_interaction_list() == ['Demag']

    # test simulation with no interaction
    # dolfin.IntervalMesh(10, 0, 1) -> dolfinx.mesh.create_interval(comm, 10, [0.0, 1.0])
    s2 = finmag.sim_with(
        mesh=mesh.create_interval(MPI.COMM_WORLD, 10, [0.0, 1.0]),
        m_init=(1, 0, 0), Ms=1,
        demag_solver=None, unit_length=1e-8)
    assert s2.get_interaction_list() == []
