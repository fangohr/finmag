import finmag
import pytest
import subprocess
import os
import sys
import numpy as np
from glob import glob
from finmag.util.visualization import render_paraview_scene
from finmag.util.visualization_impl import find_unused_X_display

# Skipping this test for now because it still doesn't work on aleph0
# (although it works on my machine) -- Max, 7.6.2013
@pytest.mark.skipif("True")
def test_render_paraview_scene(tmpdir):
    """
    This simply checks whether we can call the function. No check on
    the output image produced is performed.
    """
    tmpdir = str(tmpdir)
    os.chdir(tmpdir)

    # Check whether 'xpra' is installed
    try:
        subprocess.check_call(['xpra', '--version'])
    except OSError:
        finmag.logger.error("Could not find the 'xpra' executable, but it is needed to run this test. "
                            "Please install it using: 'sudo apt-get install xpra' (on Debian/Ubuntu-based systems).")
        sys.exit(1)

    sim = finmag.example.barmini()
    sim.save_vtk('initial_state.pvd')

    sim.schedule('save_vtk', filename='sim_relax.pvd', every=1e-10)
    sim.run_until(4.2e-10)

    display = find_unused_X_display()
    finmag.logger.debug("Rendering Paraview scene on display :{} for test.".format(display))
    subprocess.check_call(['xpra', 'start', ':{}'.format(display)])

    try:
        # XXX TODO: Maybe check various choices for all the individual arguments as well?

        # Render a single snapshots of the initial state
        render_paraview_scene('initial_state.pvd', 'initial_state.png', color_by_axis='Z',
                              rescale_colormap_to_data_range=False, debugging=False, use_display=display)
        assert(os.path.exists('initial_state.png'))

        # Render all snapshots captured during the relaxation
        render_paraview_scene('sim_relax.pvd', 'sim_relaxation_all_snapshots.png', timesteps=None, color_by_axis='Z',
                              rescale_colormap_to_data_range=False, debugging=False, use_display=display)
        assert(len(glob('sim_relaxation_all_snapshots*.png')) == 5)

        # Render only selected snapshots captured during the relaxation
        render_paraview_scene('sim_relax.pvd', 'sim_relaxation_selected_snapshots.png', timesteps=[0.0, 3.0], color_by_axis='Z',
                              rescale_colormap_to_data_range=False, debugging=False, use_display=display)
        assert(len(glob('sim_relaxation_selected_snapshots*.png')) == 2)

        # Here we test exporting timesteps that lie between the ones that are actually present in the .pvd file.
        # This works but isn't very useful, because Paraview does not interpolate the snapshots, it simply renders
        # the timesteps which are present in the .pvd file multiple times.
        render_paraview_scene('sim_relax.pvd', 'sim_relaxation_intermediate_snapshots.png', timesteps=np.linspace(0, 3.0, 10), color_by_axis='Z',
                              rescale_colormap_to_data_range=False, debugging=False, use_display=display)
        assert(len(glob('sim_relaxation_intermediate_snapshots*.png')) == 10)
    finally:
        subprocess.check_call(['xpra', 'stop', ':{}'.format(display)])
