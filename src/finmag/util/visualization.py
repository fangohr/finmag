import subprocess as sp
import textwrap
import logging
import tempfile
import shutil
import os
import sys
import IPython.core.display
from visualization_impl import _axes

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


def render_paraview_scene(
    vtu_file,
    outfile=None,
    field_name='m',
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
    trim_border=True,
    diffuse_color=None,
    debugging=False,
    use_display=None):

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
    vtu_file = os.path.abspath(vtu_file)
    if outfile is None:
        _, outfile = tempfile.mkstemp(suffix='.png')
    outfile = os.path.abspath(outfile)

    # Check that Xvfb is installed. We use it in case there is no X
    # connection (e.g. via ssh etc.)
    #
    # XXX TODO: It would be nice if we could try without Xvfb first
    #           and only rely on Xvfb if that fails.
    try:
        sp.check_output(['which', 'Xvfb'])
    except sp.CalledProcessError:
        logger.error("Xvfb is required for render_paraview_scene() to work, "
                     "but it doesn't seem to be installed or is not working "
                     "properly. Please check the installation  and try again.")
        sys.exit(1)

    #
    # Create the temporary script. The string 'script_string' will
    # contain a call to the function in 'visualization_impl.py' which
    # has all the parameter values filled in correctly.
    #
    tmpdir = tempfile.mkdtemp()
    scriptfile = os.path.join(tmpdir, 'render_scene.py')
    script_string = textwrap.dedent("""
              from visualization_impl import render_paraview_scene, find_valid_X_display
              from subprocess import Popen
              import time
              import os

              if not os.environ.has_key('DISPLAY'):
                  display = find_valid_X_display()
                  if display is None:
                      raise RuntimeError("Could not render Paraview scene as no valid X display was found.")
                  else:
                      os.environ['DISPLAY'] = ':' + str(display)
              render_paraview_scene(
                  '{}', '{}', {},
                  {}, {}, {},
                  {}, {}, {}, {},
                  '{}', {}, {},
                  '{}', {}, '{}',
                  {}, {}, {},
                  {}, {},
                  {}, '{}', '{}', {}, {})
              """.format(
            vtu_file, outfile, repr(field_name),
            camera_position, camera_focal_point, camera_view_up,
            view_size, magnification, fit_view_to_scene, color_by_axis,
            colormap, rescale_colormap_to_data_range, show_colorbar,
            colorbar_label_format, add_glyphs, glyph_type,
            glyph_scale_factor, glyph_random_mode, glyph_mask_points,
            glyph_max_number_of_points, show_orientation_axes,
            show_center_axes, representation, palette, trim_border,
            diffuse_color))
    with open(scriptfile, 'w') as f:
        f.write(script_string)
    shutil.copy(os.path.join(os.path.dirname(__file__), './visualization_impl.py'), tmpdir)

    # Execute the script in a separate process
    curdir_bak = os.getcwd()
    try:
        display_bak = os.environ['DISPLAY']
    except KeyError:
        display_bak = None
    try:
        os.chdir(tmpdir)
        if use_display is not None:
            os.environ['DISPLAY'] = ':{}'.format(use_display)
        sp.check_output(['python', 'render_scene.py'], stderr=sp.STDOUT)
    except sp.CalledProcessError as ex:
        logger.error("Could not render Paraview scene. The error message was: {}".format(ex.output))
        #raise
    finally:
        if debugging == True:
            logger.debug("Temporary directory '{}' kept for debugging. You "
                         "can try to run 'render_script.py' manually "
                         "there.".format(tmpdir))
        else:
            shutil.rmtree(tmpdir)
        os.chdir(curdir_bak)  # change back into the original directory
        if display_bak is not None:
            os.environ['DISPLAY'] = display_bak

    try:
        image = IPython.core.display.Image(filename=outfile)
    except IOError:
        # Something went wrong (missing X display?); let's not choke but return None instead.
        image = None
    return image


# Set the docstring of the wrapped function so that it reflects the
# actual implementation.

from visualization_impl import render_paraview_scene as render_scene_impl
render_paraview_scene.__doc__ = render_scene_impl.__doc__
