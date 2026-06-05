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
from dev.dolfinx.relaxation_example import main
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


def test_relaxation_parameters_reject_invalid_values():
    """Invalid prototype parameters should fail before form assembly."""
    with pytest.raises(ValueError, match="anisotropy_constant must be non-negative"):
        RelaxationParameters(anisotropy_constant=-1.0)

    with pytest.raises(ValueError, match="exchange_constant must be non-negative"):
        RelaxationParameters(exchange_constant=-1.0)

    with pytest.raises(ValueError, match="saturation_magnetisation must be positive"):
        RelaxationParameters(saturation_magnetisation=0.0)

    with pytest.raises(ValueError, match="unit_length must be positive"):
        RelaxationParameters(unit_length=0.0)

    with pytest.raises(ValueError, match="field must have three components"):
        RelaxationParameters(field=(0.0, 1.0))

    with pytest.raises(ValueError, match="anisotropy_axis must have three components"):
        RelaxationParameters(anisotropy_axis=(0.0, 1.0))

    with pytest.raises(ValueError, match="field must contain numeric values"):
        RelaxationParameters(field=("not", "numeric", "values"))

    with pytest.raises(ValueError, match="anisotropy_axis must contain numeric values"):
        RelaxationParameters(anisotropy_axis=(object(), 0.0, 1.0))


def test_relaxation_example_cli_writes_valid_json(tmp_path, monkeypatch, capsys):
    """Cover the command-line path used by the M4 verification wrapper."""
    output_path = tmp_path / "cli-summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "relaxation_example",
            "--output",
            str(output_path),
            "--steps",
            "2",
            "--dt",
            "0.01",
        ],
    )

    main()

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text())
    assert printed == written
    assert validate_summary(written) == written
    assert written["steps"] == 2


def test_relaxation_example_cli_rejects_invalid_controls(tmp_path, monkeypatch):
    """The CLI should expose the same explicit validation as the function API."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "relaxation_example",
            "--output",
            str(tmp_path / "bad-steps.json"),
            "--steps",
            "0",
        ],
    )
    with pytest.raises(ValueError, match="steps must be positive"):
        main()

    monkeypatch.setattr(
        "sys.argv",
        [
            "relaxation_example",
            "--output",
            str(tmp_path / "bad-dt.json"),
            "--dt",
            "0",
        ],
    )
    with pytest.raises(ValueError, match="dt must be positive"):
        main()


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
