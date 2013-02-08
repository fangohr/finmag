import os
import subprocess as sp
import numpy as np

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def test_against_nmag():
    cwd_backup = os.getcwd()
    os.chdir(MODULE_DIR)

    try:
        # the nmag file should be in version control. However, it is convenient
        # that the test recomputes it if needed.
        try:
            filename = "averages_ref.txt"
            sp.check_output(['make', filename], stderr=sp.STDOUT)
            m_nmag = np.genfromtxt(os.path.join(MODULE_DIR, filename))

            filename = "averages.txt"
            sp.check_output(['make', filename], stderr=sp.STDOUT)
            m_finmag = np.genfromtxt(os.path.join(MODULE_DIR, filename))
        except sp.CalledProcessError as ex:
            print("Running 'make {}' was unsuccessful. The error "
                  "message was: {}".format(filename, ex.output))
            raise

    finally:
        os.chdir(cwd_backup)

    np.testing.assert_allclose(m_nmag, m_finmag, rtol=1e-2)
    # atol is 0 by default when using assert_allclose
