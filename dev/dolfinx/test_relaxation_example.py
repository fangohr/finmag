"""Tests for the checked M4 DOLFINx relaxation example.

The example is intentionally small, but it exercises the current M4 end-to-end
path: create a mesh, set magnetisation, evaluate common energies, take explicit
LLG steps, and write basic JSON output. [Codex gpt-5.5 high]
"""

import json

import numpy as np
import pytest

from dev.dolfinx.relaxation_example import SUMMARY_SCHEMA_VERSION
from dev.dolfinx.relaxation_example import RelaxationParameters
from dev.dolfinx.relaxation_example import run_relaxation_example, validate_summary


def test_relaxation_example_writes_json_summary(tmp_path):
    """The M4 example should produce deterministic machine-readable output."""
    output_path = tmp_path / "summary.json"

    summary = run_relaxation_example(output_path, steps=5, dt=1e-2)

    written = json.loads(output_path.read_text())
    assert written == summary
    assert validate_summary(written) == written
    assert written["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert written["mesh"] == "unit_square_2x2"
    assert written["steps"] == 5
    assert len(written["energy_history"]) == 6
    assert written["final_energy"] < written["initial_energy"]
    assert np.isclose(np.linalg.norm(written["final_average_m"]), 1.0)
    assert written["final_average_m"][2] > 0.0


def test_relaxation_example_rejects_invalid_controls(tmp_path):
    """Invalid example controls should fail before producing misleading output."""
    with pytest.raises(ValueError, match="steps must be positive"):
        run_relaxation_example(tmp_path / "bad-steps.json", steps=0)

    with pytest.raises(ValueError, match="dt must be positive"):
        run_relaxation_example(tmp_path / "bad-dt.json", dt=0.0)


def test_relaxation_example_accepts_parameter_dataclass(tmp_path):
    """The prototype API shape should allow explicit parameter objects."""
    parameters = RelaxationParameters(
        anisotropy_constant=0.1,
        exchange_constant=2.0,
        saturation_magnetisation=2.0,
    )

    summary = run_relaxation_example(
        tmp_path / "custom-summary.json",
        steps=3,
        parameters=parameters,
    )

    assert summary["steps"] == 3
    assert summary["parameters"]["anisotropy_constant"] == 0.1
    assert summary["parameters"]["exchange_constant"] == 2.0
    assert summary["parameters"]["saturation_magnetisation"] == 2.0
    assert validate_summary(summary) == summary


def test_relaxation_summary_validation_rejects_broken_output(tmp_path):
    """The JSON contract should fail explicitly for malformed summaries."""
    summary = run_relaxation_example(tmp_path / "summary.json", steps=2)

    missing_key = dict(summary)
    del missing_key["energy_history"]
    with pytest.raises(ValueError, match="missing keys: energy_history"):
        validate_summary(missing_key)

    bad_history = dict(summary)
    bad_history["energy_history"] = bad_history["energy_history"][:-1]
    with pytest.raises(ValueError, match="history length"):
        validate_summary(bad_history)

    bad_energy = dict(summary)
    bad_energy["final_energy"] = bad_energy["initial_energy"]
    with pytest.raises(ValueError, match="final energy does not match"):
        validate_summary(bad_energy)
