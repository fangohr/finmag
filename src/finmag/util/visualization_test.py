import finmag
import pytest
import subprocess
import os
import sys
from finmag.util.visualization import render_paraview_scene
from finmag.util.visualization_impl import find_unused_X_display

# Skipping this test for now because it still doesn't work on aleph0
# (although it works on my machine) -- Max, 7.6.2013
@pytest.skip
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

    display = find_unused_X_display()
    finmag.logger.debug("Rendering Paraview scene on display :{} for test.".format(display))
    subprocess.check_call(['xpra', 'start', ':{}'.format(display)])

    # XXX TODO: Maybe call this function a few times with varying parameters?
    render_paraview_scene(
        'initial_state000000.vtu', 'initial_state.png',
        color_by_axis='Z', rescale_colormap_to_data_range=False, debugging=False, use_display=display)

    subprocess.check_call(['xpra', 'stop', ':{}'.format(display)])

    # Check that the expected output file exists
    assert(os.path.exists('initial_state.png'))
