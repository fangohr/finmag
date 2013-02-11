import pickle
import subprocess as sp
import textwrap
import logging
import tempfile
import shutil
import os
import IPython.core.display

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


class ColorMap(object):
    def __init__(self, color_space, rgb_points, nan_color):
        self.color_space = color_space
        self.rgb_points = rgb_points
        self.nan_color = nan_color


_color_maps = {
    "coolwarm":
        ColorMap('Diverging',
                 [0.0, 0.231373, 0.298039, 0.752941,
                  1.0, 0.705882, 0.0156863, 0.14902],
                 [0.247059, 0, 0]),

    "heated_body":
        ColorMap('RGB',
                 [0.0, 0, 0, 0,
                  0.4, 0.901961, 0, 0,
                  0.8, 0.901961, 0.901961, 0,
                  1.0, 1, 1, 1],
                 [0, 0.498039, 1]),

    "blue_to_red_rainbow":
        ColorMap('HSV',
                 [0.0, 0, 0, 1,
                  1.0, 1, 0, 0],
                 [0.498039, 0.498039, 0.498039]),
    }

_axes = {'x': 0, 'y': 1, 'z': 2, 'magnitude': -1}
_axes_names = {0: 'x', 1: 'y', 2: 'z', -1: 'magnitude'}

_representations = ['3D Glyphs', 'Outline', 'Points', 'Surface',
                    'Surface With Edges', 'Volume', 'Wireframe']

plotting_code = """
from __future__ import division

from paraview import servermanager
import paraview.simple as pv

import os
import textwrap
import logging
import subprocess
import IPython.core.display
import numpy as np

logger = logging.getLogger("finmag")


class ColorMap(object):
    def __init__(self, color_space, rgb_points, nan_color):
        self.color_space = color_space
        self.rgb_points = rgb_points
        self.nan_color = nan_color


_color_maps = {
    "coolwarm":
        ColorMap('Diverging',
                 [0.0, 0.231373, 0.298039, 0.752941,
                  1.0, 0.705882, 0.0156863, 0.14902],
                 [0.247059, 0, 0]),

    "heated_body":
        ColorMap('RGB',
                 [0.0, 0, 0, 0,
                  0.4, 0.901961, 0, 0,
                  0.8, 0.901961, 0.901961, 0,
                  1.0, 1, 1, 1],
                 [0, 0.498039, 1]),

    "blue_to_red_rainbow":
        ColorMap('HSV',
                 [0.0, 0, 0, 1,
                  1.0, 1, 0, 0],
                 [0.498039, 0.498039, 0.498039]),
    }


_axes = {'x': 0, 'y': 1, 'z': 2, 'magnitude': -1}
_axes_names = {0: 'x', 1: 'y', 2: 'z', -1: 'magnitude'}

_representations = ['3D Glyphs', 'Outline', 'Points', 'Surface',
                    'Surface With Edges', 'Volume', 'Wireframe']


def render_paraview_scene(
    vtu_file,
    outfile,
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
    trim_border=True):

    # Paraview crashes if there is no X server running, so we check
    # whether this is the case.
    if not os.environ.has_key('DISPLAY'):
        logger.warning(
            "Could not detect a running X server (this may happen, for "
            "example, if you are on a ssh connection without X forwarding; "
            "use 'ssh -X' in this case). Aborting because Paraview is "
            "likely to crash.")

    if not representation in _representations:
        raise ValueError("Unsupported representation: '{}'. Allowed values: "
                         "{}".format(representation, _representations))

    if abs(magnification - int(magnification)) > 1e-6:
        logger.warning("Due to limitations in Paraview, the 'magnification' "
                       "argument must be an integer (got: {}). Using nearest "
                       "integer value.".format(magnification))
        magnification = int(round(magnification))

    if not os.path.exists(vtu_file):
        raise IOError("vtu file '{}' does not exist.".format(vtu_file))

    servermanager.Disconnect()
    servermanager.Connect()
    reader = servermanager.sources.XMLUnstructuredGridReader(FileName=vtu_file)
    reader.UpdatePipeline()
    view = servermanager.CreateRenderView()
    repr = servermanager.CreateRepresentation(reader, view)
    repr.Representation = representation

    view.CameraPosition = camera_position
    view.CameraFocalPoint = camera_focal_point
    view.CameraViewUp = camera_view_up

    if fit_view_to_scene:
        # N.B.: this email describes a more sophisticated (= proper?)
        # way of doing this, but it's probably overkill for now:
        #
        # http://www.paraview.org/pipermail/paraview/2012-March/024352.html
        #
        view.ResetCamera()

    view.OrientationAxesVisibility = (1 if show_orientation_axes else 0)
    view.CenterAxesVisibility = (1 if show_center_axes else 0)

    if palette == 'print':
        view.Background = [1.0, 1.0, 1.0]
        view.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
        repr.CubeAxesColor = [0.0, 0.0, 0.0]
        repr.AmbientColor = [0.0, 0.0, 0.0]
    elif palette == 'screen':
        view.Background = [0.32, 0.34, 0.43]
        view.OrientationAxesLabelColor = [1.0, 1.0, 1.0]
        repr.CubeAxesColor = [1.0, 1.0, 1.0]
        repr.AmbientColor = [1.0, 1.0, 1.0]
    else:
        raise ValueError("Palette argument must be either 'print' "
                         "or 'screen'. Got: {}".format(palette))


    # Convert color_by_axis to integer and store the name separately
    try:
        color_by_axis = _axes[color_by_axis.lower()]
    except AttributeError:
        if not color_by_axis in [0, 1, 2, -1]:
            raise ValueError("color_by_axis must have one of the values "
                             "[0, 1, 2, -1] or ['x', 'y', 'z', 'magnitude']. "
                             "Got: {}".format(color_by_axis))
    color_by_axis_name = _axes_names[color_by_axis]

    dataInfo = reader.GetDataInformation()
    pointDataInfo = dataInfo.GetPointDataInformation()
    arrayInfo = pointDataInfo.GetArrayInformation(field_name)

    if rescale_colormap_to_data_range:
        data_range = arrayInfo.GetComponentRange(color_by_axis)
        logger.debug("Rescaling colormap to data range: {}".format(data_range))
    else:
        data_range = (-1.0, 1.0)

    # Set the correct colormap and rescale it if necessary.
    try:
        cmap = _color_maps[colormap]
        if colormap == 'blue_to_red_rainbow':
            print(textwrap.dedent(\"\"\"
                Use of the 'rainbow' color map is discouraged as it has a number of distinct
                disadvantages. Use at your own risk! For details see, e.g., [1], [2].

                [1] K. Moreland, "Diverging Color Maps for Scientific Visualization"
                    http://www.sandia.gov/~kmorel/documents/ColorMaps/ColorMapsExpanded.pdf

                [2] http://www.paraview.org/ParaView3/index.php/Default_Color_Map
                \"\"\"))
    except KeyError:
        raise ValueError("Unsupported colormap: '{}'. Allowed values: "
                         "{}".format(colormap, _color_maps.keys()))
    lut = servermanager.rendering.PVLookupTable()
    lut.ColorSpace = cmap.color_space
    rgb_points = cmap.rgb_points
    dmin, dmax = data_range
    cmin = rgb_points[0]
    cmax = rgb_points[-4]
    if cmin == cmax:
        # workaround for the case that the data range only
        # contains a single value
        cmax += 1e-8
    for i in xrange(0, len(rgb_points), 4):
        rgb_points[i] = (rgb_points[i] - cmin) / (cmax - cmin) * (dmax - dmin) + dmin
    lut.RGBPoints = rgb_points
    lut.NanColor = cmap.nan_color

    if color_by_axis in [0, 1, 2]:
        lut.VectorMode = "Component"
        lut.VectorComponent = color_by_axis
    elif color_by_axis == -1:
        lut.VectorMode = "Magnitude"
        lut.VectorComponent = color_by_axis
    repr.LookupTable = lut
    repr.ColorArrayName = field_name
    repr.ColorAttributeType = "POINT_DATA"

    if add_glyphs:
        logger.debug("Adding cone glyphs.")
        glyph = pv.servermanager.filters.Glyph(Input=reader)
        if glyph_scale_factor == None:
            # Try to determine an appropriate scale_factor automatically
            import vtk.util.numpy_support as VN
            grid = servermanager.Fetch(reader)

            # Determine approximate mesh spacing
            cell = grid.GetCell(0)  # let's hope that the first cell is a good
                                    # representative of all mesh cells
            cell_bounds = np.array(cell.GetBounds()).reshape((3,2))
            mesh_spacing = float(min(cell_bounds[:, 1] - cell_bounds[:, 0]))

            # Determine maximum field magnitude
            m = VN.vtk_to_numpy(grid.GetPointData().GetArray(field_name))
            max_field_magnitude = float(max(map(np.linalg.norm, m)))

            glyph_scale_factor = mesh_spacing / max_field_magnitude
            logger.debug(
                "Using automatically determined glyph_scale_factor = {:.2g} "
                "(determined from approximate mesh spacing {:.2g} and maximum "
                "field magnitude {:.2g}). This may need manual tweaking in case "
                "glyphs appear very large or very small.".format(
                        glyph_scale_factor, mesh_spacing, max_field_magnitude))

        glyph.SetScaleFactor = glyph_scale_factor
        glyph.ScaleMode = 'vector'
        glyph.Vectors = ['POINTS', field_name]
        try:
            glyph.KeepRandomPoints = 1  # only relevant for animation IIUC, but can't hurt setting it
        except AttributeError:
            # Older version of Paraview which doesn't support this setting. Ignoring for now.
            pass
        glyph.RandomMode = glyph_random_mode
        glyph.MaskPoints = glyph_mask_points
        glyph.MaximumNumberofPoints = glyph_max_number_of_points

        if glyph_type != 'cones':
            glyph_type = 'cones'
            logger.warning("Unsupported glyph type: '{}'. "
                          "Falling back to 'cones'.".format(glyph_type))

        if glyph_type == 'cones':
            cone = servermanager.sources.Cone()
            cone.Resolution = 20
            cone.Radius = 0.2
        else:
            # This should not happen as we're catching it above.
            raise NotImplementedError()

        glyph.SetPropertyWithName('Source', cone)
        glyph_repr = servermanager.CreateRepresentation(glyph, view)
        glyph_repr.LookupTable = lut
        glyph_repr.ColorArrayName = 'GlyphVector'
        glyph_repr.ColorAttributeType = "POINT_DATA"

    if show_colorbar:
        # XXX TODO: Remove the import of paraview.simple once I know why
        from paraview.simple import CreateScalarBar
        scalarbar = CreateScalarBar(
            Title=field_name, ComponentTitle=color_by_axis_name.capitalize(),
            Enabled=1, LabelFontSize=12, TitleFontSize=12)
        scalarbar.LabelFormat = colorbar_label_format,
        if palette == 'print':
            scalarbar.LabelColor = [0.0, 0.0, 0.0]  # black labels for print
        else:
            scalarbar.LabelColor = [1.0, 1.0, 1.0]  # white labels for screen
        view.Representations.append(scalarbar)
        scalarbar.LookupTable = lut

    reader.UpdatePipelineInformation()

    view.ViewSize = view_size
    view.WriteImage(outfile, "vtkPNGWriter", magnification)
    servermanager.Disconnect()
    if trim_border:
        if palette == 'print':
            bordercolor = '"rgb(255,255,255)"'
        else:
            bordercolor = '"rgb(82,87,110)"'
        cmd = 'mogrify -bordercolor {} -border 1x1 -trim {}'.format(bordercolor, outfile)
        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
            logger.debug("Trimming border from rendered scene.")
        except OSError:
            logger.warning("Using the 'trim' argument requires ImageMagick to be installed.")
        except subprocess.CalledProcessError as ex:
            logger.warning("Could not trim border from image. "
                           "The error message was: {}".format(ex.output))

    image = IPython.core.display.Image(filename=outfile)
    return image
"""


def render_paraview_scene(
    vtu_file,
    outfile,
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
    trim_border=True):
    """
    Load a *.vtu file, render the scene in it and save the result to an image file.


    *Returns*

    An IPython.core.display.Image object containing the output image.


    *Arguments*

    vtu_file:

        Input filename (must be in *.vtu format).

    outfile:

        Name of the output image file. The image type (e.g. PNG) is
        derived from the file extension.

    field_name:

        The field to plot. Default: 'm' (= the normalised magnetisation).
        Note that this field must of course have been saved in the .vtu
        file.

    camera_position:  3-vector
    camera_focal_point:  3-vector
    camera_view_up:  3-vector

        These variables control the position and orientation of the
        camera.

    view_size: pair of int

        Controls the size of the view. This can be used to adjust the
        size and aspect ratio of the visible scene (useful for example
        if a colorbar is present). Default: (400, 400).

    magnification: int

        Magnification factor which controls the size of the saved image.
        Note that due to limitations in Paraview this must be an integer.

    fit_view_to_scene: True | False

        If True (the default), the view is automatically adjusted so
        that the entire scene is visible. In this case the exact
        location of the camera is ignored and only its relative
        position w.r.t. the focal point is taken into account.

    color_by_axis: integer or string (allowed values: 0, 1, 2, -1,
                                      or 'x', 'y', 'z', 'magnitude')

        The vector components in the direction of this axis are used
        to color the plot. If '-1' is given, the vector magnitudes
        are used instead of any vector components.

    colormap:

        The colormap to use. Supported values:
        {}.

    rescale_colormap_to_data_range:  True | False

        If False (default: True), the colormap corresponds to the data
        range [-1.0, +1.0]. If set to True, the colormap is rescaled
        so that it corresponds to the minimum/maximum data values.

    show_colorbar: True | False

        If True (the default), a colorbar is added to the plot.

    colorbar_label_format: string

        Controls how colorbar labels are formatted (e.g., how many
        digits are displayed, etc.). This can be any formatting string
        for floating point numbers as understood by Python's 'print'
        statement. Default: '%-#5.2g'.

    add_glyphs: True | False

        If True (the default), little glyphs are added at the mesh
        vertices to indicate the direction of the field vectors.

    glyph_type: string

        Type of glyphs to use. The only currently supported glyph type
        is 'cones'.

    glyph_scale_factor: float | None

        Controls the glyph size. If None is given (the default), the function
        tries to determine an appropriate scale factor automatically (based
        on the maximum field values and the mesh spacing. This may not be
        perfect, however, and may need manual adjusting.

    glyph_mask_points: True | False

        If True (the default), limit the maximum number of glyphs to
        the value indicated by glyph_max_number_of_points.

    glyph_max_number_of_points: int

        Specifies the maximum number of glyphs that should appear in
        the output dataset if glyph_mask_points is True.

    glyph_random_mode: True | False

        If True (the default), the glyph positions are chosen
        randomly. Otherwise the point IDs to which glyphs are attached
        are evenly spaced. This setting only has an effect if
        glyph_mask_points is True.

    show_orientation_axes: False | True

        If True (default: False), a set of three small axes is added
        to the scene to indicate the directions of the coordinate axes.

    show_center_axes: False | True

        If True (default: False), a set of three axes is plotted at
        the center of rotation.

    representation: string

        Controls the way in which the visual representation of bodies
        in the scene. Allowed values:
        {}

    palette:  'print' | 'screen'

        The color scheme to be used. The main difference is that
        'print' uses white as the background color whereas 'screen'
        uses dark grey.

    trim_border: True | False

        If True (the default), any superfluous space around the scene
        will be trimmed from the saved image. This requires imagemagick
        to be installed.

    """

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
    outfile = os.path.abspath(outfile)

    #
    # Create the temporary script
    #
    tmpdir = tempfile.mkdtemp()
    scriptfile = os.path.join(tmpdir, 'render_scene.py')
    script_string = textwrap.dedent("""
              from visualization_impl import render_paraview_scene

              render_paraview_scene(
                  '{}', '{}', '{}',
                  {}, {}, {},
                  {}, {}, {}, {},
                  '{}', {}, {},
                  '{}', {}, '{}',
                  {}, {}, {},
                  {}, {},
                  {}, '{}', '{}', {})
              """.format(
            vtu_file, outfile, field_name,
            camera_position, camera_focal_point, camera_view_up,
            view_size, magnification, fit_view_to_scene, color_by_axis,
            colormap, rescale_colormap_to_data_range, show_colorbar,
            colorbar_label_format, add_glyphs, glyph_type,
            glyph_scale_factor, glyph_random_mode, glyph_mask_points,
            glyph_max_number_of_points, show_orientation_axes,
            show_center_axes, representation, palette, trim_border))
    with open(scriptfile, 'w') as f:
        f.write(script_string)
    with open(os.path.join(tmpdir, 'visualization_impl.py'), 'w') as f:
        f.write(plotting_code)

    # Execute the script in a separate process
    curdir_bak = os.getcwd()
    try:
        os.chdir(tmpdir)
        with open('/dev/null') as devnull:
            sp.check_output(['python', 'render_scene.py'] , stderr=devnull)
    except sp.CalledProcessError as ex:
        logger.error("Could not render Paraview scene. The error "
                     "message was: {}".format(ex.output))
        raise
    finally:
        shutil.rmtree(tmpdir)
        os.chdir(curdir_bak)  # change back into the original directory

    image = IPython.core.display.Image(filename=outfile)
    return image

# Automatically add supported colormaps and representations to the docstring:
render_paraview_scene.__doc__ = \
    render_paraview_scene.__doc__.format(_color_maps.keys(), _representations)
