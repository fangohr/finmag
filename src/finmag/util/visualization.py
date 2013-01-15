import logging
logger = logging.getLogger("finmag")
logger.warning("This module will probably crash when imported from within Finmag, "
               "but the code does work on its own. There seems to be some kind of "
               "weird incompability which needs to be fixed (although I have no "
               "idea what's causing it).")
import os
import IPython
from paraview import servermanager

_color_maps = {
    "coolwarm": [-1.0, 0.23, 0.299, 0.754,
                  1.0, 0.706, 0.016, 0.15],
    }

_axes = {'x': 0, 'y': 1, 'z': 2, 'magnitude': -1}
_axes_names = {0: 'x', 1: 'y', 2: 'z', -1: 'magnitude'}

def render_paraview_scene(vtu_file, outfile, color_by_axis=0,
                          rescale_to_data_range=False,
                          colormap='blue_to_red', show_colorbar=False,
                          colorbar_label_format="%-#5.2g",
                          show_orientation_axes=False,
                          show_center_axes=False,
                          field_name='m'):
    """
    Load a *.vtu file, render the scene in it and save the result to a
    file.

    *Returns*

    An IPython.core.display.Image object containing the output image.


    *Arguments*

    vtu_file:

        Input filename (must be in *.vtu format).

    outfile:

        Name of the output image file. The type (e.g. png) is derived
        from the file extension.

    color_by_axis: integer (allowed values: 0, 1, 2 and -1)

        Use the vector components in the direction of this axis to
        color the plot. If '-1' is given, the vector *magnitudes* are
        used.

    rescale_to_data_range: [False | True]

        If False (the default), the colormap corresponds to the data
        range [-1.0, +1.0]. If set to True, the colormap is rescaled
        so that it corresponds to the minimum/maximum data values.

    colormap:

        The colormap to use. Supported values: {}.
    """


    if not os.path.exists(vtu_file):
        raise IOError("vtu file '{}' does not exist.".format(vtu_file))
    servermanager.Disconnect()
    servermanager.Connect()
    reader = servermanager.sources.XMLUnstructuredGridReader(FileName=vtu_file)
    reader.UpdatePipeline()
    view = servermanager.CreateRenderView()
    repr = servermanager.CreateRepresentation(reader, view)
    repr.Representation = "Surface With Edges"
    view.ResetCamera()
    view.OrientationAxesVisibility = (1 if show_orientation_axes else 0)
    view.CenterAxesVisibility = (1 if show_center_axes else 0)


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
        raise ValueError("Unsupported color map: {}. Allowed values: "
                         "{}".format(colormap, _color_maps.keys()))
    if rescale_to_data_range:
        print("Rescaling colormap to data range.")
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
            ComponentTitle=color_by_axis_name.capitalize(),
            Title=field_name, Enabled=1, LabelFontSize=12, TitleFontSize=12)
        scalarbar.LabelFormat = colorbar_label_format,
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


# Automatically add all supported colormaps to the docstring:
render_paraview_scene.__doc__ = render_paraview_scene.__doc__.format(_color_maps.keys())
