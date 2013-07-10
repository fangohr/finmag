import dolfin as df
import numpy as np
import os
from glob import glob
from finmag import sim_with
from finmag.example import barmini
from finmag.util.helpers import plot_hysteresis_loop

ONE_DEGREE_PER_NS = 17453292.5 # in rad/s

H = 0.2e6  # maximum external field strength in A/m
initial_direction = np.array([1.0, 0.01, 0.0])
N = 5


def test_hysteresis(tmpdir):
    os.chdir(str(tmpdir))
    sim = barmini()
    mesh = df.BoxMesh(0, 0, 0, 1, 1, 1, 1, 1, 1)
    H_ext_list = [(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)]
    N = len(H_ext_list)

    # Run a relaxation and save a vtk snapshot at the end of each stage;
    # this should result in three .vtu files (one for each stage).
    sim1 = sim_with(mesh, Ms=1e6, m_init=(0.8, 0.2, 0), alpha=1.0,
                    unit_length=1e-9, A=None, demag_solver=None)
    sim1.schedule('save_vtk', at_end=True, filename='barmini_hysteresis.pvd')
    res1 = sim1.hysteresis(H_ext_list=H_ext_list)
    assert(len(glob('barmini_hysteresis*.vtu')) == N)
    assert(res1 == None)

    # Run a relaxation with a non-trivial `fun` argument and check
    # that we get a list of return values.
    sim2 = sim_with(mesh, Ms=1e6, m_init=(0.8, 0.2, 0), alpha=1.0,
                    unit_length=1e-9, A=None, demag_solver=None)
    res2 = sim2.hysteresis(H_ext_list=H_ext_list,
                           fun=lambda sim: sim.m_average[0])
    assert(len(res2) == N)


def test_hysteresis_loop_and_plotting(tmpdir):
    """
    Call the hysteresis loop with various combinations for saving
    snapshots and check that the correct number of vtk files have been
    produced. Also check that calling the plotting function works
    (although the output image isn't verified).

    """
    os.chdir(str(tmpdir))

    mesh = df.BoxMesh(0, 0, 0, 1, 1, 1, 1, 1, 1)
    sim = sim_with(mesh, Ms=1e6, m_init=(0.8, 0.2, 0), alpha=1.0,
                   unit_length=1e-9, A=None, demag_solver=None)
    H_vals, m_vals = \
        sim.hysteresis_loop(H, initial_direction, N, stopping_dmdt=10)

    # Check that the magnetisation values are as trivial as we expect
    # them to be ;-)
    assert(np.allclose(m_vals, [1.0 for _ in xrange(2*N)], atol=1e-4))

    # This only tests whether the plotting function works without
    # errors. It currently does *not* check that it produces
    # meaningful results (and the plot is quite boring for the system
    # above anyway).
    plot_hysteresis_loop(H_vals, m_vals, infobox=["param_A = 23", ("param_B", 42)],
                         title="Hysteresis plot test", xlabel="H_ext", ylabel="m_avg",
                         figsize=(5,4), infobox_loc="bottom left",
                         filename='test_plot.pdf')

    # Test multiple filenames, too
    plot_hysteresis_loop(H_vals, m_vals, filename=['test_plot.pdf', 'test_plot.png'])
