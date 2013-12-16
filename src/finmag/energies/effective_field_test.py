import numpy as np
import dolfin as df
import pytest
import os
from finmag.example import barmini


def test_effective_field_compute_returns_copy(tmpdir):
    """
    Regression test to ensure that the value returned by
    EffectiveField.compute() does not change as a simulation
    progresses. This used to happen since EffectiveField.compute()
    returned a reference to an internal numpy array instead of a copy.
    Here we check that this is fixed.

    """
    os.chdir(str(tmpdir))
    sim = barmini()
    h0 = sim.llg.effective_field.compute()
    h0_copy = h0.copy()
    sim.run_until(1e-12)
    h1 = sim.llg.effective_field.compute()

    assert np.allclose(h0, h0_copy, atol=0, rtol=1e-8)
    assert not np.allclose(h0, h1, atol=0, rtol=1e-8)
