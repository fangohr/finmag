"""Tests for the checked M4 DOLFINx relaxation example.

The example is intentionally small, but it exercises the current M4 end-to-end
path: create a mesh, set magnetisation, evaluate common energies, take explicit
LLG steps, and write basic JSON output. [Codex gpt-5.5 high]
"""

import json

import numpy as np
import pytest

from dev.dolfinx.relaxation_example import run_relaxation_example


def test_relaxation_example_writes_json_summary(tmp_path):
    """The M4 example should produce deterministic machine-readable output."""
    output_path = tmp_path / "summary.json"

    summary = run_relaxation_example(output_path, steps=5, dt=1e-2)

    written = json.loads(output_path.read_text())
    assert written == summary
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
