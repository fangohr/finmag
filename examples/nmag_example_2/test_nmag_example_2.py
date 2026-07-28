import os
import pytest
import subprocess as sp
import numpy as np
try:  # master: import run_finmag
    import run_finmag
except ImportError:
    run_finmag = None  # not ported (D33): tests below xfail
from finmag.util.fileio import Tablereader

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.slow
@pytest.mark.not_ported
@pytest.mark.xfail(reason="not ported: nmag_example_2 (Task 30 deferred example, requires the not-ported OOMMF/Nmag comparison harness) (D33)", strict=True)
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

        filename = 'finmag_bar.ndt'
        if not os.path.exists(filename):
            run_finmag.run_simulation()
        f = Tablereader(os.path.join(MODULE_DIR, filename))
        m_finmag = np.array(f['time', 'm_x', 'm_y', 'm_z']).T

    except sp.CalledProcessError as ex:
        print("Running command '{}' was unsuccessful. The error "
              "message was: {}".format(cmd, ex.output))
        raise

    finally:
        os.chdir(cwd_backup)

    assert max(map(np.linalg.norm, m_nmag - m_finmag)) < 1.1e-4
    # atol is 0 by default when using assert_allclose
