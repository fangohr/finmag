import dolfin as df
import numpy as np
import os
from glob import glob
from finmag import sim_with
from finmag.util.helpers import plot_hysteresis_loop

ONE_DEGREE_PER_NS = 17453292.5 # in rad/s

H = 0.2e6  # maximum external field strength in A/m
initial_direction = np.array([1.0, 0.01, 0.0])
N = 5

def run_hysteresis_loop(save_every, save_at_stage_end, filename):
    mesh = df.BoxMesh(0, 0, 0, 1, 1, 1, 1, 1, 1)
    sim = sim_with(mesh, Ms=1e6, m_init=(0.8, 0.2, 0), alpha=1.0,
                   unit_length=1e-9, A=None, demag_solver=None)
    H_vals, m_vals = \
        sim.hysteresis_loop(H, initial_direction, N, stopping_dmdt=10,
                            save_every=save_every,
                            save_at_stage_end=save_at_stage_end,
                            filename=filename)
    return H_vals, m_vals

def test_hysteresis_loop_and_plotting(tmpdir):
    """
    Call the hysteresis loop with various combinations for saving
    snapshots and check that the correct number of vtk files have been
    produced. Also check that calling the plotting function works
    (although the output image isn't verified).

    """
    os.chdir(str(tmpdir))

    # 1) Don't save any snapshots. No files should be created.
    run_hysteresis_loop(save_every=None, save_at_stage_end=False, filename='test_01/no_snapshots.pvd')
    assert(not os.path.exists('test_01'))  # without saving, no files should exist

    # 2) Only save snapshots at the end of stages. This should result
    #    in one snapshot for each of the 2*N stages.
    run_hysteresis_loop(save_every=None, save_at_stage_end=True, filename='test_02/snapshots.pvd')
    vtu_files = sorted(glob('test_02/*.vtu'))
    vtu_files_expected = ['test_02/snapshots__stage_{:03d}__000000.vtu'.format(i) for i in xrange(2*N)]
    assert(vtu_files == vtu_files_expected)
    pvd_files = sorted(glob('test_02/*.pvd'))
    pvd_files_expected = ['test_02/snapshots.pvd'] + ['test_02/snapshots__stage_{:03d}.pvd'.format(i) for i in xrange(2*N)]
    assert(pvd_files == pvd_files_expected)

    # 3) Save snapshots at regular intervals (every 10 ns) during the
    #    simulation. Currently the first relaxation finishes at ca. 22
    #    ns, so this should result in three .vtu snapshots for the
    #    first stage. Since nothing happens at subsequent stages,
    #    there should be a single file for each of the remaining ones.
    run_hysteresis_loop(save_every=1e-10, save_at_stage_end=False, filename='test_03/snapshots.pvd')
    vtu_files = sorted(glob('test_03/*.vtu'))
    vtu_files_expected = sorted(['test_03/snapshots__stage_{:03d}__000000.vtu'.format(i) for i in xrange(2*N)] +
                                ['test_03/snapshots__stage_000__000001.vtu',
                                 'test_03/snapshots__stage_000__000002.vtu'])
    assert(vtu_files == vtu_files_expected)
    pvd_files = sorted(glob('test_03/*.pvd'))
    pvd_files_expected = sorted(['test_03/snapshots_all.pvd'] + ['test_03/snapshots__stage_{:03d}.pvd'.format(i) for i in xrange(2*N)])
    assert(pvd_files == pvd_files_expected)

    # 4) Save snapshots at regular intervals *and* at the end of each
    #    relaxation. This should add one snapshot per stage as
    #    compared to the previous test.
    H_vals, m_vals = run_hysteresis_loop(save_every=1e-10, save_at_stage_end=True, filename='test_04/snapshots.pvd')
    vtu_files = sorted(glob('test_04/*.vtu'))
    vtu_files_expected = sorted(['test_04/snapshots__stage_{:03d}__00000{}.vtu'.format(i, j) for i in xrange(2*N) for j in [0,1]] +
                                ['test_04/snapshots__stage_000__000002.vtu',
                                 'test_04/snapshots__stage_000__000003.vtu'])
    assert(vtu_files == vtu_files_expected)
    pvd_files = sorted(glob('test_04/*.pvd'))
    pvd_files_expected = sorted(['test_04/snapshots.pvd', 'test_04/snapshots_all.pvd'] + ['test_04/snapshots__stage_{:03d}.pvd'.format(i) for i in xrange(2*N)])
    assert(pvd_files == pvd_files_expected)

    # Check that the magnetisation values are as trivial as we expect
    # them to be ;-)
    assert(np.allclose(m_vals, [1.0 for _ in xrange(2*N)], atol=1e-4))

    # This only tests whether the plotting function works without
    # errors. It currently does *not* check that it produces
    # meaningful results (and the plot is quite boring for the system
    # above anyway).
    plot_hysteresis_loop(H_vals, m_vals, infobox=["param_A = 23", ("param_B", 42)], title="Hysteresis plot test",
                         xlabel="H_ext", ylabel="m_avg", figsize=(5,4), infobox_loc="bottom left")
