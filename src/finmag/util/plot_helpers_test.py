import os
import pytest
from finmag.example import barmini
from finmag.util.plot_helpers import *


@pytest.mark.requires_X_display
def test_plot_ndt_columns_and_plot_dynamics(tmpdir):
    """
    Simply check that we can call the command `plot_ndt_columns` with some arguments
    """
    os.chdir(str(tmpdir))
    sim = barmini()
    sim.schedule('save_ndt', every=1e-12)
    sim.run_until(1e-11)
    plot_ndt_columns('barmini.ndt', columns=['m_x', 'm_y', 'm_z', 'E_Demag', 'H_Exchange_x'],
                     outfile='barmini.png', title="Some awesome title",
                     show_legend=True, legend_loc='center', figsize=(10, 4))

    plot_dynamics('barmini.ndt', components='xz',
                  outfile='barmini2.png', xlim=(0, 0.8e-11), ylim=(-1, 1))

    assert(os.path.exists('barmini.png'))
    assert(os.path.exists('barmini2.png'))


