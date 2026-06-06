"""Tests for the reduced M5 DOLFINx simulation wrapper.

These tests deliberately exercise a tiny API surface. The goal is to make the
first M5 core simulation path explicit without pretending that legacy
``finmag.Simulation`` has been ported. [Codex gpt-5.5 high]
"""

import json

import numpy as np
import pytest

from dev.dolfinx.relaxation_example import RelaxationParameters
from dev.dolfinx.relaxation_example import validate_summary
from dev.dolfinx.simulation import PrototypeSimulation


def test_prototype_simulation_reports_supported_energy_terms():
    """The reduced simulation wrapper should compose the checked M4 energies."""
    sim = PrototypeSimulation.unit_square()

    terms = sim.energy_terms()

    assert set(terms) == {"anisotropy", "exchange", "total", "zeeman"}
    assert np.isclose(terms["exchange"], 0.0)
    assert np.isclose(terms["total"], terms["anisotropy"] + terms["zeeman"])


def test_prototype_simulation_state_summary_does_not_advance():
    """State summaries should expose current state without mutating it."""
    sim = PrototypeSimulation.unit_square()

    before = sim.state_summary()
    after = sim.state_summary()

    assert before == after
    assert before["mesh"] == "unit_square_2x2"
    assert set(before["energy_terms"]) == {"anisotropy", "exchange", "total", "zeeman"}
    assert np.allclose(before["average_m"], (1.0, 0.0, 0.0))


def test_prototype_simulation_trace_record_does_not_advance():
    """Trace records should be read-only snapshots of the current state."""
    sim = PrototypeSimulation.unit_square()

    before = sim.trace_record(step=0, time=0.0)
    after = sim.trace_record(step=0, time=0.0)

    assert before == after
    assert before["step"] == 0
    assert before["time"] == 0.0
    assert np.allclose(before["average_m"], (1.0, 0.0, 0.0))


def test_prototype_simulation_relaxation_decreases_energy():
    """The wrapper should expose the checked explicit relaxation workflow."""
    sim = PrototypeSimulation.unit_square()

    energy_history = sim.relax(steps=5, dt=1e-2)

    assert len(energy_history) == 6
    assert energy_history[-1] < energy_history[0]
    assert np.isclose(np.linalg.norm(sim.average_m()), 1.0)
    assert sim.average_m()[2] > 0.0


def test_prototype_simulation_accepts_custom_parameters():
    """Custom reduced parameters should be reflected in reported energies."""
    parameters = RelaxationParameters(
        anisotropy_constant=0.0,
        saturation_magnetisation=2.0,
    )
    sim = PrototypeSimulation.unit_square(parameters=parameters)

    terms = sim.energy_terms()

    assert np.isclose(terms["anisotropy"], 0.0)
    assert np.isclose(terms["zeeman"], 0.0)
    assert np.isclose(terms["total"], 0.0)


def test_prototype_simulation_rejects_invalid_controls():
    """Invalid wrapper controls should fail before creating misleading results."""
    with pytest.raises(ValueError, match="mesh dimensions must be positive"):
        PrototypeSimulation.unit_square(nx=0)

    sim = PrototypeSimulation.unit_square()
    with pytest.raises(ValueError, match="steps must be positive"):
        sim.relax(steps=0, dt=1e-2)
    with pytest.raises(ValueError, match="steps must be positive"):
        sim.relaxation_trace(steps=0, dt=1e-2)
    with pytest.raises(ValueError, match="dt must be positive"):
        sim.relaxation_trace(steps=1, dt=0.0)


def test_prototype_simulation_relaxation_summary_is_json_compatible():
    """The M5 wrapper should expose a small machine-readable output path."""
    sim = PrototypeSimulation.unit_square()

    summary = sim.relaxation_summary(steps=3, dt=1e-2)

    assert summary["mesh"] == "unit_square_2x2"
    assert summary["steps"] == 3
    assert len(summary["energy_history"]) == 4
    assert summary["final_energy"] < summary["initial_energy"]
    assert np.isclose(np.linalg.norm(summary["final_average_m"]), 1.0)
    assert summary["schema_version"] == 1
    assert validate_summary(summary) == summary


def test_prototype_simulation_writes_relaxation_summary(tmp_path):
    """The M5 wrapper should write the same validated JSON summary it returns."""
    output_path = tmp_path / "simulation-summary.json"
    sim = PrototypeSimulation.unit_square()

    summary = sim.write_relaxation_summary(output_path, steps=3, dt=1e-2)

    written = json.loads(output_path.read_text())
    assert written == summary
    assert validate_summary(written) == written


def test_prototype_simulation_relaxation_trace_is_json_compatible():
    """The M5 wrapper should expose per-step state records for data I/O."""
    sim = PrototypeSimulation.unit_square()

    trace = sim.relaxation_trace(steps=3, dt=1e-2)

    assert trace["mesh"] == "unit_square_2x2"
    assert trace["schema_version"] == 1
    assert trace["steps"] == 3
    assert len(trace["records"]) == 4
    assert [record["step"] for record in trace["records"]] == [0, 1, 2, 3]
    assert [record["time"] for record in trace["records"]] == pytest.approx(
        [0.0, 1e-2, 2e-2, 3e-2]
    )
    energies = [record["energy_terms"]["total"] for record in trace["records"]]
    assert energies[-1] < energies[0]
    assert np.isclose(np.linalg.norm(trace["records"][-1]["average_m"]), 1.0)


def test_prototype_simulation_writes_relaxation_trace(tmp_path):
    """The M5 wrapper should write the same JSON trace it returns."""
    output_path = tmp_path / "simulation-trace.json"
    sim = PrototypeSimulation.unit_square()

    trace = sim.write_relaxation_trace(output_path, steps=2, dt=1e-2)

    written = json.loads(output_path.read_text())
    assert written == trace
    assert len(written["records"]) == 3


def test_prototype_simulation_restart_state_round_trips(tmp_path):
    """Reduced restart JSON should recreate the same prototype state."""
    output_path = tmp_path / "simulation-restart.json"
    sim = PrototypeSimulation.unit_square()
    sim.relax(steps=2, dt=1e-2)

    state = sim.write_restart_state(output_path)
    restarted = PrototypeSimulation.read_restart_state(output_path)

    assert json.loads(output_path.read_text()) == state
    assert state["schema_version"] == 1
    assert state["mesh"] == "unit_square_2x2"
    assert np.allclose(restarted.average_m(), sim.average_m())
    assert restarted.parameters == sim.parameters
    assert restarted.energy_terms() == pytest.approx(sim.energy_terms())


def test_prototype_simulation_restart_state_rejects_invalid_input():
    """Malformed reduced restart states should fail before state mutation."""
    state = PrototypeSimulation.unit_square().restart_state()

    wrong_schema = dict(state, schema_version=999)
    with pytest.raises(ValueError, match="unsupported restart schema"):
        PrototypeSimulation.from_restart_state(wrong_schema)

    unsupported_mesh = dict(state, mesh="custom")
    with pytest.raises(ValueError, match="only unit-square restart states"):
        PrototypeSimulation.from_restart_state(unsupported_mesh)

    wrong_values = dict(state, magnetisation_values=[[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="magnetisation shape"):
        PrototypeSimulation.from_restart_state(wrong_values)

    missing_parameter = dict(state["parameters"])
    del missing_parameter["field"]
    with pytest.raises(ValueError, match="restart parameters are missing keys"):
        PrototypeSimulation.from_restart_state(dict(state, parameters=missing_parameter))
