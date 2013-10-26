# This file contains the actual implementation of the Paraview-based
# plotting code. Unfortunately, we cannot import this directly into
# Finmag due to an incompatibility between the Python modules
# 'paraview.servermanager' and 'vtk' which can lead to segfaults (see
# comment in visualization.py). Instead, we need to call this as an
# external process (using the subprocess module) in order to avoid the
# conflict.
#
# NOTE: If you make any changes in the signature (or docstring) of
#       this function, make sure to make the corresponding changes in
#       visualization.py. (both in the signature of 'render_paraview_scene'
#       and in the definition of 'script_string').

from __future__ import division

import os
import sh
import textwrap
import tempfile
import logging
import IPython.core.display
import subprocess as sp
import numpy as np
import numbers
import shlex
from threading import Timer


logger = logging.getLogger("finmag")


def find_valid_X_display(displays_to_try=xrange(100)):
    """
    Sequentially checks all X displays in the given list (default: 0 through 99)
    and returns the number of the first valid display that is found. Returns None
    if no valid display could be found.

    *Arguments*

    displays_to_try:  list of displays to search (default: [0, ..., 99])

    """
    # A (probably faster) alternative way would be to write a temporary
    # shell script which contains the loop and run that script using a
    # single subprocess call. However, since usually display :0 will be
    # available the loop below should terminate quite quickly.
    for display in displays_to_try:
        try:
            sp.check_output(['xdpyinfo', '-display', ':{}'.format(display)], stderr=sp.STDOUT)
            # This display is available since the command finished successfully
            logger.debug("Found valid display :{}".format(display))
            return display
        except sp.CalledProcessError:
            # This display is not available
            continue
    logger.debug("No valid display found.")
    return None


def find_unused_X_display(displays_to_try=xrange(100)):
    """
    Sequentially checks all X displays in the given list (default: 0 through 99)
    and returns the number of the first unused display that is found. Returns None
    if no unused display could be found.

    *Arguments*

    displays_to_try:  list of displays to search (default: [0, ..., 99])

    """
    for display in displays_to_try:
        try:
            sp.check_output(['xdpyinfo', '-display', ':{}'.format(display)], stderr=sp.STDOUT)
            # This display is already in used since the command finished successfully
        except sp.CalledProcessError:
            logger.debug("Found unused display :{}".format(display))
            return display
    logger.debug("No unused display found.")
    return None


def has_passwordless_ssh_login(hostname, timeout_sec=5):
    """
    Check whether it is possible to login to the given host without specifying a password.
    Will give up after `timeout_sec` seconds.

    """
    def run_cmd(cmd, timeout_sec):
        proc = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.PIPE)
        kill_proc = lambda p: p.kill()
        timer = Timer(timeout_sec, kill_proc, [proc])
        timer.start()
        stdout, stderr = proc.communicate()
        timer.cancel()
        return proc.returncode

    returncode= run_cmd('ssh -oNumberOfPasswordPrompts=0 {} "echo hello"'.format(hostname), timeout_sec=timeout_sec)
    return (returncode == 0)


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
    pvd_file,
    outfile,
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
    diffuse_color=None):
    """
    Load a *.vtu file, render the scene in it and save the result to an image file.


    *Returns*

    An IPython.core.display.Image object containing the output image.


    *Arguments*

    pvd_file:

        Input filename (must be in *.pvd format).

    outfile:

        Name of the output image file (may be None, which is the
        default). The image type (e.g. PNG) is derived from the file
        extension. If multiple timesteps are to be animated, the
        output files will have additional suffixes of the form
        '_N_TIMESTEP', where N represents the index of the timestep
        (in the array passed as the argument `timesteps`) and TIMESTEP
        is the actual timestep itself.

    field_name:

        The field to plot. Default: 'm' (= the normalised magnetisation).
        Note that this field must of course have been saved in the .vtu
        file.

    timesteps:

        The timesteps for which to render the scene. The default is
        None, which means to animate all timesteps (and save them as a
        sequence of images if `outfile` is specified). Other valid
        values are either a single number or a list of numbers.

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
        so that it corresponds to the minimum/maximum data values *over
        all specified timesteps*.

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

    use_parallel_projection: True | False

        If False (the default), perspective projection is used to
        render the scene. Otherwise parallel projection is used.

    trim_border: True | False

        If True (the default), any superfluous space around the scene
        will be trimmed from the saved image. This requires imagemagick
        to be installed.

    rescale: float | None

        Factor by which the output image will be rescaled. For example,
        using 'rescale=0.4' will rescale the image by 40%.

    diffuse_color: 3-tuple of RGB values

        The solid color of the body. If given, this overrides any
        colormap-related values.

    """
    from paraview import servermanager
    import paraview.simple as pv

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

    if not os.path.exists(pvd_file):
        raise IOError("File does not exist: '{}'.".format(pvd_file))

    servermanager.Disconnect()
    servermanager.Connect()
    reader = servermanager.sources.PVDReader(FileName=pvd_file)
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

    view.CameraParallelProjection = 1 if use_parallel_projection else 0

    # Convert color_by_axis to integer and store the name separately
    try:
        color_by_axis = _axes[color_by_axis.lower()]
    except AttributeError:
        if not color_by_axis in [0, 1, 2, -1]:
            raise ValueError("color_by_axis must have one of the values "
                             "[0, 1, 2, -1] or ['x', 'y', 'z', 'magnitude']. "
                             "Got: {}".format(color_by_axis))
    color_by_axis_name = _axes_names[color_by_axis]

    if timesteps is None:
        timesteps = reader.TimestepValues
    elif not isinstance(timesteps, (list, tuple, np.ndarray)):
        if not isinstance(timesteps, numbers.Number):
            raise TypeError("Argument 'timesteps' must be either None or a number or a list of numbers. Got: '{}'".format(timesteps))
        timesteps = [timesteps]

    data_range = (-1.0, 1.0)
    if rescale_colormap_to_data_range:
        dmin, dmax = np.infty, -np.infty
        for t in timesteps:
            reader.UpdatePipeline(t)
            dataInfo = reader.GetDataInformation()
            pointDataInfo = dataInfo.GetPointDataInformation()
            arrayInfo = pointDataInfo.GetArrayInformation(field_name)
            cur_data_range = arrayInfo.GetComponentRange(color_by_axis)
            dmin = min(cur_data_range[0], dmin)
            dmax = max(cur_data_range[1], dmax)
        data_range = (dmin, dmax)
        logger.debug("Rescaling colormap to data range: {}".format(data_range))


    # Set the correct colormap and rescale it if necessary.
    try:
        cmap = _color_maps[colormap]
        if colormap == 'blue_to_red_rainbow':
            print(textwrap.dedent("""
                Use of the 'rainbow' color map is discouraged as it has a number of distinct
                disadvantages. Use at your own risk! For details see, e.g., [1], [2].

                [1] K. Moreland, "Diverging Color Maps for Scientific Visualization"
                    http://www.sandia.gov/~kmorel/documents/ColorMaps/ColorMapsExpanded.pdf

                [2] http://www.paraview.org/ParaView3/index.php/Default_Color_Map
                """))
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
    if diffuse_color is not None:
        print "diffuse_color: {} ({})".format(diffuse_color, type(diffuse_color))
        repr.DiffuseColor = diffuse_color
    else:
        repr.LookupTable = lut
    if field_name is not None:
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

    if outfile is None:
        _, outfile = tempfile.mkstemp(suffix='.png')

    view.ViewSize = view_size

    def write_image(outfilename):
        _, suffix = os.path.splitext(outfilename)
        if suffix == '.png':
            view.WriteImage(outfilename, "vtkPNGWriter", magnification)
        elif suffix in ['.jpg', '.jpeg']:
            view.WriteImage(outfilename, "vtkJPEGWriter", magnification)
        else:
            raise ValueError("Output image must have extension '.png' or "
                             "'.jpg'. Got: {}".format(suffix))
        if trim_border:
            if palette == 'print':
                bordercolor = '"rgb(255,255,255)"'
            else:
                # Strangely, we get a slightly different background
                # color for PNG than for JPG.
                bordercolor = '"rgb(82,87,110)"' if (suffix == '.png') else '"rgb(82,87,109)"'
            cmd = 'mogrify -bordercolor {} -border 1x1 -trim {}'.format(bordercolor, outfilename)
            try:
                sp.check_output(cmd, stderr=sp.STDOUT, shell=True)
                logger.debug("Trimming border from rendered scene.")
            except OSError:
                logger.warning("Using the 'trim' argument requires ImageMagick to be installed.")
            except sp.CalledProcessError as ex:
                logger.warning("Could not trim border from image. "
                               "The error message was: {}".format(ex.output))

        if rescale:
            rescale_factor = int(rescale * 100.0)
            cmd = 'mogrify -resize {:d}% {}'.format(rescale_factor, outfilename)
            try:
                sp.check_output(cmd, stderr=sp.STDOUT, shell=True)
                logger.debug("Resizing output image by {:d}%".format(rescale_factor))
            except OSError:
                logger.warning("Using the 'rescale' argument requires ImageMagick to be installed.")
            except sp.CalledProcessError as ex:
                logger.warning("Could not rescale image. The error message was: {}".format(ex.output))

    if len(timesteps) == 1:
        # If a single timestep is rendered, we return the resulting image.
        view.ViewTime = timesteps[0]
        write_image(outfile)
        res = IPython.core.display.Image(filename=outfile)
    else:
        # Otherwise we export a bunch of images with sequentially
        # numbered suffixes.
        #
        # TODO: What should we return? Just the image for the first
        #       timestep as we currently do? Or can we somehow create
        #       a video and return that?
        outbasename, outsuffix = os.path.splitext(outfile)

        def generate_outfilename(i, t):
            return outbasename + '_{:04d}_'.format(i) + str(t) + outsuffix

        for (i, t) in enumerate(timesteps):
            view.ViewTime = t
            cur_outfilename = generate_outfilename(i, t)
            logger.debug("Saving timestep {} to file '{}'.".format(t, cur_outfilename))
            write_image(cur_outfilename)
        res = IPython.core.display.Image(filename=generate_outfilename(0, timesteps[0]))

    servermanager.Disconnect()

    return None

# Automatically add supported colormaps and representations to the docstring:
render_paraview_scene.__doc__ = \
    render_paraview_scene.__doc__.format(_color_maps.keys(), _representations)
