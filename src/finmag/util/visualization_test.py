import finmag
import pytest
import os
from finmag.util.visualization import render_paraview_scene

def test_render_paraview_scene(tmpdir):
    """
    This simply checks whether we can call the function. No check on
    the output image produced is performed.
    """
    tmpdir = str(tmpdir)
    os.chdir(tmpdir)
    sim = finmag.example.barmini()
    sim.save_vtk('initial_state.pvd')

    # XXX TODO: Maybe call this function a few times with varying parameters?
    render_paraview_scene(
        'initial_state000000.vtu', 'initial_state.png',
        color_by_axis='Z', rescale_colormap_to_data_range=False, debugging=False)
