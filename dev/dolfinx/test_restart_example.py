"""Tests for the M5 DOLFINx restart-state round-trip example.

The restart example is intentionally small: it proves the reduced
``PrototypeSimulation`` JSON restart state can be used as an executable witness
without introducing a general Finmag restart format. [Codex gpt-5.5 high]
"""

import json

import pytest

from dev.dolfinx.restart_example import run_restart_example


def test_restart_example_round_trips_in_memory():
    """The example should prove restart equivalence without writing a file."""
    summary = run_restart_example(output_path=None)

    assert summary["mesh"] == "unit_square_2x2"
    assert summary["restart_path"] is None
    assert summary["restart_schema_version"] == 1
    assert summary["roundtrip_matches"]
    assert summary["max_average_m_error"] == 0.0
    assert summary["max_energy_error"] == 0.0


def test_restart_example_round_trips_via_json_file(tmp_path):
    """The example should also exercise the persisted JSON restart state."""
    output_path = tmp_path / "restart-state.json"

    summary = run_restart_example(output_path=output_path, steps=3, dt=1e-2)

    written = json.loads(output_path.read_text())
    assert written["mesh"] == "unit_square_2x2"
    assert written["schema_version"] == 1
    assert summary["restart_path"] == str(output_path)
    assert summary["steps"] == 3
    assert summary["roundtrip_matches"]


def test_restart_example_rejects_invalid_controls():
    """Invalid example controls should fail before producing artefacts."""
    with pytest.raises(ValueError, match="steps must be positive"):
        run_restart_example(steps=0)

    with pytest.raises(ValueError, match="dt must be positive"):
        run_restart_example(dt=0.0)
