import os
import subprocess
import numpy as np

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def test_against_nmag():
    cwd_backup = os.getcwd()
    os.chdir(MODULE_DIR)

    try:
        # the nmag file should be in version control. However, it is convenient
        # that the test recomputes it if needed.
        subprocess.call("make averages_ref.txt", shell=True)
        m_nmag = np.genfromtxt(os.path.join(MODULE_DIR, "averages_ref.txt"))

        subprocess.call("make averages.txt", shell=True)
        m_finmag = np.genfromtxt(os.path.join(MODULE_DIR, "averages.txt"))

    finally:
        os.chdir(cwd_backup)

    np.testing.assert_allclose(m_nmag, m_finmag, rtol=1e-2)
    # atol is 0 by default when using assert_allclose
