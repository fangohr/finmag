import numpy as np
from run_finmag import run_simulation

def test_against_nmag():
    TOLERANCE = 2e-5

    # this file should be in version control
    m_nmag = np.genfromtxt("averages_ref.txt")

    run_simulation()
    m_finmag = np.genfromtxt("averages.txt")

    diff = np.abs(m_nmag - m_finmag)
    assert np.max(diff) < TOLERANCE
