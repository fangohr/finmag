import finmag
import pytest
import subprocess
import os
import sys
import numpy as np
import dolfin as df
from glob import glob
from finmag.util.visualization import *
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


def test_flight_path_rotation():
    # Random starting position and axis
    p0 = [-1, 3, 2]
    axis = [2.4, 5.2, 8.0]

    f = flight_path_rotation(start_pos=p0, axis=axis, angle=360)

    # A few sanity checks:

    # For a full rotation, the inital and end points should coincide with p0.
    assert(np.allclose(f(0), p0))
    assert(np.allclose(f(1), p0))

    # All connecting vectors between the initial position and any
    # other should be orthogonal to the rotation axis.
    for t in np.linspace(0, 1, 100):
        v = f(t) - f(0)
        assert(abs(np.dot(v, axis)) < 1e-8)

    # If the starting position lies on the axis then all points on the
    # flight path should coincide.
    f = flight_path_rotation(start_pos=p0, axis=p0, angle=42.0)
    for t in np.linspace(0, 1, 100):
        assert(np.allclose(f(t), p0))


def test_flight_path_straight_line():
    # Random starting position and axis
    P0 = np.array([-1, 3, 2])
    P1 = np.array([2.4, 5.2, 8.0])

    f = flight_path_straight_line(P0, P1)

    # The start and end point should coincide with P0 and P1
    assert(np.allclose(f(0), P0))
    assert(np.allclose(f(1), P1))

    # Compare f(t) with an explicit linear interpolation between P0 and P1
    t_vals = np.linspace(0, 1, 200)

    for t in t_vals:
        pt_expected = (1 - t) * P0 + t * P1
        assert(np.allclose(f(t), pt_expected))


@pytest.mark.skipif("True")
def test_plot_dolfin_function(tmpdir):
    os.chdir(str(tmpdir))
    interval_mesh = df.UnitIntervalMesh(2)
    square_mesh = df.UnitSquareMesh(2, 2)
    cube_mesh = df.UnitCubeMesh(2, 2, 2)

    S = df.FunctionSpace(cube_mesh, 'CG', 1)
    V2 = df.VectorFunctionSpace(square_mesh, 'CG', 1, dim=3)
    V3 = df.VectorFunctionSpace(cube_mesh, 'CG', 1, dim=3)

    s = df.Function(S)
    v2 = df.Function(V2)
    v3 = df.Function(V3); v3.vector()[:] = 1.0

    # Wrong function space dimension
    with pytest.raises(TypeError):
        plot_dolfin_function(s, outfile='buggy.png')

    # Plotting a 3D function on a 3D mesh should work
    plot_dolfin_function(v3, outfile='plot.png')
    assert(os.path.exists('plot.png'))

    # Try 2-dimensional mesh as well
    plot_dolfin_function(v2, outfile='plot_2d_mesh.png')
    assert(os.path.exists('plot_2d_mesh.png'))
