import os
import subprocess as sp
import numpy as np
import run_finmag

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def test_against_nmag():
    cwd_backup = os.getcwd()
    os.chdir(MODULE_DIR)

    try:
        cmd = ['make', 'clean']
        sp.call(cmd)

        # the nmag file should be in version control. However, it is
        # convenient that the test can recompute it if needed.
        filename = 'averages_ref.txt'
        cmd = ['make', filename]
        sp.call(cmd)
        m_nmag = np.genfromtxt(os.path.join(MODULE_DIR, filename))

        if not os.path.exists("averages.txt"):
            run_finmag.run_simulation()
            cmd = 'ncol finmag_bar.ndt time m_x m_y m_z > averages.txt'
            sp.call(cmd, shell=True)
        m_finmag = np.genfromtxt(os.path.join(MODULE_DIR, 'averages.txt'))

    except sp.CalledProcessError as ex:
        print("Running command '{}' was unsuccessful. The error "
              "message was: {}".format(cmd, ex.output))
        raise

    finally:
        os.chdir(cwd_backup)

    assert max(map(np.linalg.norm, m_nmag - m_finmag)) < 3e-5
    # atol is 0 by default when using assert_allclose
