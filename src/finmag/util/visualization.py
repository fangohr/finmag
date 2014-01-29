from __future__ import division
import sh
import numpy as np
import textwrap
import logging
import tempfile
import shutil
import sys
import os
import re
import dolfin as df
import IPython.core.display
from glob import glob
from visualization_impl import *
from visualization_impl import _axes
from math import sin, cos, pi

logger = logging.getLogger("finmag")

# Unfortunately, there is an incompatibility between the Python
# modules 'paraview.servermanager' and 'vtk' which can cause a
# segfault if the vtk module is imported first. Since we cannot
# control whether this is done by the user (e.g. implicitly by
# importing dolfin), the workaround for now is to save a mini-script
# with the Paraview rendering command to a temporary file and execute
# that in a separate process.
#
# The actual plotting code is contained in the string 'plotting_code'
# below. Note that some of the code in there is intentionally
# duplicated in the rest of this file. This is so that once the
# segfault issue is resolved we can simply remove everything else and
# only take the code from the string, and it should hopefully work
# without changes.

# XXX TODO: The function 'find_valid_X_display' should ideally be
# defined in this module, but this isn't possible due to the paraview
# incompatibilities mentioned above. To avoid code duplication and
# errors due to not keeping the two in sync, we only define it in
# visualization_impl.py and import it here.
from visualization_impl import find_valid_X_display
from finmag.util import configuration


def flight_path_rotation(start_pos, axis=[0, 0, 1], angle=360):
    """
    Return a function `f(t)` which defines a 'flight path' of a
    rotating camera at time `t` (where `t` runs from 0 to 1).

    *Arguments*

    start_pos:

        Starting position of the camera at time t=0.

    axis:

        Rotation axis. Default: [0, 0, 1] (i.e., the z-axis)

    angle:

        The angle (in degrees) of the entire rotating motion.
        Default: 360 (= one full rotation).
    """
    start_pos = np.asarray(start_pos)
    axis_normalised = np.asarray(axis / np.linalg.norm(axis))
    angle_rad = angle * pi / 180.

    # Find the radial component of the starting position vector
    r1 = start_pos - np.dot(start_pos, axis_normalised) * axis_normalised

    # P0 is the 'anchor point' on the rotation axis
    P0 = start_pos - r1

    # Find another vector orthogonal to both the axis and to r1 (of
    # the same length as r1). Together, r1 and r2 define the rotation
    # plane.
    r2 = np.cross(axis_normalised, r1)

    print "P0: {}".format(P0)
    print "r1: {}".format(r1)
    print "r2: {}".format(r2)

    def flight_path(t):
        pos = P0 + cos(t*angle_rad) * r1 + sin(t*angle_rad) * r2
        return pos

    return flight_path


def flight_path_straight_line(start_pos, end_pos):
    """
    Return a function `f(t)` which defines a 'flight path' of a camera
    moving along a straight line between `start_pos` and `end_pos`
    (where `t` runs from 0 to 1).

    """
    start_pos = np.asarray(start_pos)
    end_pos = np.asarray(end_pos)

    def flight_path(t):
        return (1 - t) * start_pos + t * end_pos

    return flight_path


def render_paraview_scene(
    pvd_file,
    outfile=None,
    field_name='m',
    timesteps=None,
    camera_position=[0, -200, +200],
    camera_focal_point=[0, 0, 0],
    camera_view_up=[0, 0, 1],
    view_size=(800, 600),
    magnification=1,
    fit_view_to_scene=True,
    color_by_axis=0,
    colormap='coolwarm',
    rescale_colormap_to_data_range=True,
    show_colorbar=True,
    colorbar_label_format='%-#5.2g',
    add_glyphs=True,
    glyph_type='cones',
    glyph_scale_factor=None,
    glyph_random_mode=True,
    glyph_mask_points=True,
    glyph_max_number_of_points=10000,
    show_orientation_axes=False,
    show_center_axes=False,
    representation="Surface With Edges",
    palette='screen',
    use_parallel_projection=False,
    trim_border=True,
    rescale=None,
    diffuse_color=None,
    debug=False,
    use_display=None,
    hostname=None):

    # Convert color_by_axis to integer and store the name separately
    try:
        color_by_axis = _axes[color_by_axis.lower()]
    except AttributeError:
        if not color_by_axis in [0, 1, 2, -1]:
            raise ValueError("color_by_axis must have one of the values "
                             "[0, 1, 2, -1] or ['x', 'y', 'z', 'magnitude']. "
                             "Got: {}".format(color_by_axis))

    # Use absolute path for filenames because the script will be
    # written to a temporary directory in a different location.
    pvd_file = os.path.abspath(pvd_file)
    if outfile is None:
        _, outfile = tempfile.mkstemp(suffix='.png')
        outfile_is_temporary = True
    else:
        outfile_is_temporary = False
    outfile = os.path.abspath(outfile)

    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):
        logger.debug("Creating non-existing directory component '{}' of output filename.".format(outdir))
        os.makedirs(outdir)
        logger.debug("Done.")

    #
    # Create the temporary script. The string 'script_string' will
    # contain a call to the function in 'visualization_impl.py' which
    # has all the parameter values filled in correctly.
    #
    tmpdir = create_tmpdir_on_host(hostname)
    scriptfile = os.path.join(tmpdir, 'render_scene.py')
    pvd_basename = os.path.basename(pvd_file)
    tmp_pvdfile = os.path.join(tmpdir, pvd_basename)
    tmp_outfile = os.path.join(tmpdir, 'tmp_outfile' + os.path.splitext(outfile)[1])
    copy_file_to_host(None, pvd_file, hostname, tmpdir)
    for vtu_file in glob(os.path.splitext(pvd_file)[0] + '*.vtu'):
        copy_file_to_host(None, vtu_file, hostname, tmpdir)
    script_string = textwrap.dedent("""
              from visualization_impl import render_paraview_scene, find_valid_X_display
              from subprocess import Popen
              from numpy import array
              import time
              import os

              if not os.environ.has_key('DISPLAY'):
                  display = find_valid_X_display()
                  if display is None:
                      raise RuntimeError("Could not render Paraview scene as no valid X display was found.")
                  else:
                      os.environ['DISPLAY'] = ':' + str(display)
              render_paraview_scene(
                  '{}', '{}', {}, {},
                  {}, {}, {},
                  {}, {}, {}, {},
                  '{}', {}, {},
                  '{}', {}, '{}',
                  {}, {}, {},
                  {}, {},
                  {}, '{}', '{}', {},
                  {}, {}, {})
              """.format(
            tmp_pvdfile, tmp_outfile, repr(field_name), re.sub('\n', '', repr(timesteps)),
            camera_position, camera_focal_point, camera_view_up,
            view_size, magnification, fit_view_to_scene, color_by_axis,
            colormap, rescale_colormap_to_data_range, show_colorbar,
            colorbar_label_format, add_glyphs, glyph_type,
            glyph_scale_factor, glyph_random_mode, glyph_mask_points,
            glyph_max_number_of_points, show_orientation_axes,
            show_center_axes, representation, palette, use_parallel_projection,
            trim_border, rescale, diffuse_color))
    write_file_on_host(hostname, scriptfile, script_string)
    copy_file_to_host(None, os.path.join(os.path.dirname(__file__), './visualization_impl.py'),
                      hostname, tmpdir)

    # Execute the script in a separate process
    curdir_bak = os.getcwd()
    xpra_display = None
    use_xpra = configuration.get_config_option("visualization", "use_xpra", "True")
    try:
        display_bak = os.environ['DISPLAY']
    except KeyError:
        display_bak = None
    try:
        #os.chdir(tmpdir)

        if use_display is None and use_xpra.lower() != "false":
            # Try to create a display using 'xpra'
            try:
                # Check whether 'xpra' is installed
                run_command_on_host(hostname, 'xpra', '--version')
                xpra_display = find_unused_X_display(xrange(1, 100), hostname=hostname)
                run_command_on_host(hostname, 'xpra', 'start', ':{}'.format(xpra_display))
                use_display = xpra_display
                logger.debug("Rendering Paraview scene on display :{} using xpra.".format(xpra_display))
            except sh.CommandNotFound:
                logger.warning(
                    "Could not find the 'xpra' executable. You may want to "
                    "install it to avoid annoying pop-up windows from "
                    "Paraview. Under Debian/Ubuntu you can install it via "
                    "'sudo apt-get install xpra'.")
                xpra_display = None

        if use_display is not None:
            os.environ['DISPLAY'] = ':{}'.format(use_display)

        run_command_on_host(hostname, 'python', scriptfile)
        copy_file_to_host(hostname, tmp_outfile, None, outfile)
    except sh.ErrorReturnCode as ex:
        logger.error("Could not render Paraview scene. The error message was: {}".format(ex.message))
        #raise
    finally:
        if debug == True:
            logger.debug("Temporary directory '{}' kept for debugging. You "
                         "can try to run 'render_script.py' manually "
                         "there.".format(tmpdir))
        else:
            #shutil.rmtree(tmpdir)
            remove_tmpdir_on_host(hostname, tmpdir)
        os.chdir(curdir_bak)  # change back into the original directory

        if xpra_display is not None:
            # XXX TODO: It may be nice to keep the xpra display open
            #           until Finmag exits, because we are likely to
            #           render more than one snapshot.
            run_command_on_host(hostname, 'xpra', 'stop', ':{}'.format(xpra_display))

        if display_bak is not None:
            os.environ['DISPLAY'] = display_bak
        else:
            os.environ.pop('DISPLAY', None)

    try:
        image = IPython.core.display.Image(filename=outfile)
    except IOError:
        # Something went wrong (missing X display?); let's not choke but return None instead.
        image = None

    if outfile_is_temporary:
        # Clean up temporary file
        os.remove(outfile)

    return image


def plot_dolfin_function(f, **kwargs):
    """
    Uses Paraview to plot the given dolfin Function (currently this
    only works for a dolfin.Function representing a 3D vector field).
    Returns an IPython.display.Image object containing the rendered
    scene. All keyword arguments are passed on to the function
    `finmag.util.visualization.render_paraview_scene`, which is used
    internally.

    """
    # Check that f represents a 3D vector field defined on a 3D mesh.
    if not f.element().value_shape() == (3,):
        raise TypeError("The function to be plotted must represent a 3D vector field.")

    f.rename('f', 'f')
    tmpdir = tempfile.mkdtemp()
    tmpfilename = os.path.join(tmpdir, 'dolfin_function.pvd')
    try:
        # Save the function to a temporary file
        funcfile = df.File(tmpfilename)
        funcfile << f
        kwargs.pop('field_name', None)  # ignore this argument as we are using our own field_name
        return render_paraview_scene(tmpfilename, field_name='f', **kwargs)
    finally:
        shutil.rmtree(tmpdir)



# Set the docstring of the wrapped function so that it reflects the
# actual implementation.

from visualization_impl import render_paraview_scene as render_scene_impl
render_paraview_scene.__doc__ = render_scene_impl.__doc__
