import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from collections import Iterable
import numpy as np

def plot_m(sim, component='x', filename=None, figsize=(10, 5),
          extent=None, z=0.0, gridpoints=[100, 100], cbar=True, ncbarticks=5,
          cmap='RdBu', bgcolor='w', scale_by_Ms=False, axis_units='nm'):
    """
    Plotting function for the magnetisation.
    Inputs
    ------
    sim: simulation object of type Simulation or NormalModeSimulation

    component: str
        'x', 'y', 'z', 'all' or 'angle'

    filename: str (None)
        File to save the figure to - does not save if not specified.

    figsize: 2-tuple
        Matplotlib figure size specifiers in inches.
        If not 'all', the x-size is 1/3 of what is specified,
        such that consistency in size is maintained between
        'all' and other plots.

    extent: None, float, or 4-tuple
        If None, the bounds are calculated from the
        finite element mesh.

        If a single number, the calculated extent is
        multiplied by that - e.g. if extent=1.1, if the
        mesh bounds are [-50, 50, -50, 50] then the new
        bounds are [-55, 55, -55, 55].

        If specified, directly, must be in
        the format:

        [-ve x bounds of plot, +ve x-bounds of plot,
         -ve y bounds of plot, +ve y-bounds of plot]

    z: float
        Height at which to sample the field.

    gridpoints: 2-tuple of integers
        [Number of gridpoints in x, gridpoints in y]
        Because this is finite elements, we can sample
        at arbitrary places. However, sampling can
        be fairly costly, so the higher the number of
        samples, the longer plotting will take.

    cbar: Boolean
        Plot a colorbar, or not...

    ncbarticks:
        Number of values listed on colorbar axis.
        Ignored if no colorbar.

    cmap:
        Matplotlib colormap name. For magnetisation,
        divergent colormaps, like RdBu, tend to work
        well.

        For spin angles, a cyclic map like 'hsv' works
        better.
        See the full list here:

        https://matplotlib.org/examples/color/colormaps_reference.html

    bcolor: str
        Color specifier for background. Areas outside of
        the mesh are set to this color. 'w' for white,
        'k' for black, or use a hexadecimal color code,
        or a tuple of RGB values.

    axis_units: str
        Units for the x and y axis labels, in case
        nm does not make sense.
    """

    # If the user doesn't specify plot bounds,
    # calculate them from the mesh.
    if not isinstance(extent, Iterable):
        xs = sim.mesh.coordinates()[:, 0]
        ys = sim.mesh.coordinates()[:, 1]
        xmin = np.min(xs)
        xmax = np.max(xs)
        ymin = np.min(ys)
        ymax = np.max(ys)
        calc_bounds = np.array([xmin, xmax, ymin, ymax])
        if type(extent) is float:
            extent *= calc_bounds
        else:
            extent = calc_bounds

    components = ['x', 'y', 'z']

    x = np.linspace(extent[0], extent[1], gridpoints[0])
    y = np.linspace(extent[2], extent[3], gridpoints[1])
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    Z = np.zeros_like(X) + z

    mx = []
    my = []
    mz = []
    Ms = []

    for xv, yv, zv in zip(X, Y, Z):
        try:
            # Sample the field at the mesh.
            mxv, myv, mzv = sim.llg.m_field.f(xv, yv, zv)
            if scale_by_Ms is True:
                msv = sim.Ms((xv, yv, zv))

        except RuntimeError:
            # Set nan's for places outside of the mesh,
            # as this allows us to set the colormap
            # for bad values and hence set a background color
            # not in the colorrange of a given colormap.
            mxv, myv, mzv = np.nan, np.nan, np.nan
            msv = 0.0
            # Note - nan * 0.0 = nan,
            # so this is fine and doesn't mess with our
            # bounds later!

        mx.append(mxv)
        my.append(myv)
        mz.append(mzv)
        Ms.append(msv)

    mx = np.array(mx).reshape(gridpoints[0], gridpoints[1])
    my = np.array(my).reshape(gridpoints[0], gridpoints[1])
    mz = np.array(mz).reshape(gridpoints[0], gridpoints[1])
    Ms = np.array(Ms).reshape(gridpoints[0], gridpoints[1])

    if scale_by_Ms is True:
        mx *= Ms
        my *= Ms
        mz *= Ms

    m = [mx, my, mz]

    if component in ['x', 'y', 'z', 'angle']:
        fig = plt.figure(figsize=(figsize[0]/3, figsize[1]))
        # Have to use ImageGrid in order to get the Colorbar
        # to scale in size with the subplots!
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 1),
                         axes_pad=0.15,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.15,
                         )
        ax = grid[0]
        ax.set_xlabel('$x$ ({})'.format(axis_units))
        ax.set_ylabel('$y$ ({})'.format(axis_units))

        # Note: By default, imshow plots like a matrix rather
        # than a Cartesian axis, and so below we have to set
        # origin = 'lower' everywhere.

        if component is 'angle':
            theta = np.arctan2(my, mx)
            theta[theta < 0] += 2*np.pi

            cmap_edited = plt.get_cmap(cmap)
            # Edit the colormap and set bad values to bgcolor
            cmap_edited.set_bad(color=bgcolor, alpha=1.0)
            # Here we set the bounds between 0 and 2*pi for the angle,
            # though there's no reason why it couldn't be -pi and pi
            # really.
            im = ax.imshow(theta, origin='lower',
                           extent=extent, vmin=0,
                           vmax=2*np.pi, cmap=cmap_edited)
            ax.set_title('$xy$ angle')

        else:
            cmap_edited = plt.get_cmap(cmap)
            cmap_edited.set_bad(color=bgcolor, alpha=1.0)
            if scale_by_Ms is True:
                vmin = -np.max(Ms)
                vmax = np.max(Ms)
            else:
                vmin = -1.0
                vmax = 1.0

            im = ax.imshow(m[components.index(component)], origin='lower',
                           extent=extent, vmin=vmin, vmax=vmax,
                           cmap=cmap_edited)
            ax.set_title('$m_{}$'.format(component))

    elif component is 'all':
        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 3),
                         axes_pad=0.15,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.15,
                         )

        if scale_by_Ms is True:
            vmin = -np.max(Ms)
            vmax = np.max(Ms)
        else:
            vmin = -1.0
            vmax = 1.0

        for ax, comp, label in zip(grid, [mx, my, mz], ['x', 'y', 'z']):
            im = ax.imshow(comp, origin='lower', extent=extent,
                           vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_title('$m_{}$'.format(label))
            ax.set_xlabel('$x$ ({})'.format(axis_units))
            ax.set_ylabel('$y$ ({})'.format(axis_units))

    else:
        raise ValueError("Component is not valid")

    if cbar is True:
        if component is 'angle':
            # Some special handling to print \pi
            # rather than the numbers!
            cbar = ax.cax.colorbar(im,
                                   ticks=np.linspace(0, 2*np.pi, ncbarticks))

            cbarlabels = ['${:.1f} \pi$'.format(x/(np.pi))
                          if x != 0.0 else '0.0'
                          for x in np.linspace(0, 2*np.pi, ncbarticks)]
            cbar.ax.set_yticklabels(cbarlabels)

        else:
            cbar = ax.cax.colorbar(im,
                                   ticks=np.linspace(vmin, vmax, ncbarticks))

        if scale_by_Ms:
            cbar.ax.set_ylabel('A / m', rotation=0)
        ax.cax.toggle_label(True)

    if filename:
        fig.savefig(filename, dpi=1000)
