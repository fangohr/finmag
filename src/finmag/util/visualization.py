import logging
logger = logging.getLogger("finmag")
logger.warning("This module will probably crash when imported from within "
               "Finmag, but the code does work on its own. There seems to be "
               "some kind of weird incompability which needs to be fixed "
               "(although I have no idea what could be causing it).")
import os
import IPython.core.display
from paraview import servermanager

_color_maps = {
    "coolwarm": [-1.0, 0.23, 0.299, 0.754,
                  1.0, 0.706, 0.016, 0.15],
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
    fit_view_to_scene=True,
    color_by_axis=0,
    colormap='blue_to_red',
    rescale_colormap_to_data_range=False,
    show_colorbar=False,
    colorbar_label_format='%-#5.2g',
    show_orientation_axes=False,
    show_center_axes=False,
    representation="Surface With Edges",
    palette='screen'):
    """
    Load a *.vtu file, render the scene in it and save the result to
    an image file.


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

    rescale_colormap_to_data_range:  False | True

        If False (the default), the colormap corresponds to the data
        range [-1.0, +1.0]. If set to True, the colormap is rescaled
        so that it corresponds to the minimum/maximum data values.

    colormap:

        The colormap to use. Supported values:
        {}.

    show_colorbar: False | True

        If True (default: False), a colorbar is added to the plot.

    colorbar_label_format: string

        Controls how colorbar labels are formatted (e.g., how many
        digits are displayed, etc.). This can be any formatting string
        for floating point numbers as understood by Python's 'print'
        statement. Default: '%-#5.2g'.

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
    """
    if not representation in _representations:
        raise ValueError("Unsupported representation: '{}'. Allowed values: "
                         "{}".format(representation, _representations))


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
    view.CameraClippingRange = [111.82803847855743, 244.8030589195661]  # Paraview's default values; don't know what exactly they mean

    if fit_view_to_scene:
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
    data_range = arrayInfo.GetComponentRange(color_by_axis)
    logger.debug("Data range: {}".format(data_range))

    # Set the correct colormap and rescale it if necessary.
    lut = servermanager.rendering.PVLookupTable()
    try:
        rgb_points = _color_maps[colormap]
    except KeyError:
        raise ValueError("Unsupported colormap: '{}'. Allowed values: "
                         "{}".format(colormap, _color_maps.keys()))
    if rescale_colormap_to_data_range:
        logger.debug("Rescaling colormap to data range.")
        rgb_points[0] = data_range[0]
        rgb_points[4] = data_range[1]
    lut.RGBPoints = rgb_points

    if color_by_axis in [0, 1, 2]:
        lut.VectorMode = "Component"
        lut.VectorComponent = color_by_axis
    elif color_by_axis == -1:
        lut.VectorMode = "Magnitude"
        lut.VectorComponent = color_by_axis
    repr.LookupTable = lut
    repr.ColorArrayName = field_name
    repr.ColorAttributeType = "POINT_DATA"

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
    
    # XXX TODO: we put this import here because at toplevel it seems
    # to connect to a servermanager, which causes problems. Would be
    # good to figure out how to avoid using paraview.simple (and also
    # how to avoid that a separate window opens during WriteImage, but
    # that's only a minor annoyance).
    #import paraview.simple as pv
    #pv.WriteImage(outfile, view=view)
    view.WriteImage(outfile, "vtkPNGWriter", 1)
    servermanager.Disconnect()
    image = IPython.core.display.Image(filename=outfile)
    return image


# Automatically add supported colormaps and representations to the docstring:
render_paraview_scene.__doc__ = \
    render_paraview_scene.__doc__.format(_color_maps.keys(), _representations)
