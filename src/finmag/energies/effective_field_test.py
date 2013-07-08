import numpy as np
from finmag.example import barmini


def test_unexpected_behaviour():
    """
    This test documents some potentially unexpected behaviour, where
    the value of a variable which was assigned the return value of
    sim.llg.effective_field.compute() can change without prior notice.
    This is because currently sim.llg.effective_field.compute()
    returns a 'reference' to its internal numpy array which can get
    updated as the simulation progresses.

    This is done for efficiency, but it can cause great confusion if
    the user isn't aware of it. Maybe we should change it in the
    future, but for now this test documents the current behaviour.

    """
    sim = barmini()
    h0 = sim.llg.effective_field.compute()
    h0_copy = h0.copy()
    sim.run_until(1e-12)
    h1 = sim.llg.effective_field.compute()

    # Yes, these assert statements are correct. The behaviour is *not*
    # what you would expect.
    assert not np.allclose(h0, h0_copy, atol=0, rtol=1e-8)
    assert (h0 == h1).all()
