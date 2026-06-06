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
