"""
Easy way to plot values of a function R2 -> R in 2D or 3D.

Does not offer much flexibility, but can be helpful for quick visualisation
of data in an ipython notebook for instance.

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# don't let the pyflakes error "unused import" in the next line fool you
from mpl_toolkits.mplot3d import axes3d  # used in fig.gca(projection="3d")
from finmag.util.fileio import Tablereader
import types
from .helpers import *

def surface_2d(x, y, u, labels=("", "", ""), title="",
               ylim=None, xlim=None, clim=None, cmap=cm.coolwarm, path="", **kwargs):
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)

    X, Y = np.meshgrid(x, y)
    surf = ax.pcolormesh(X, Y, u, cmap=cmap)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if clim is not None:
        surf.set_clim(vmin=clim[0], vmax=clim[1])
    cb = fig.colorbar(surf)
    cb.ax.yaxis.set_label_position('right')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    cb.ax.set_ylabel(labels[2])

    plt.title(title)
    plt.savefig(path) if path else plt.show()


def surface_3d(x, y, u, labels=("", "", ""), title="", path=""):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, u, cmap=cm.coolwarm)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    plt.title(title)
    plt.savefig(path) if path else plt.show()


def quiver(f, mesh, filename=None, title="", **kwargs):
    """
    Takes a numpy array of the values of a vector-valued function, defined
    over a mesh (either a dolfin mesh, or one from finmag.util.oommf.mesh)
    and shows a quiver plot of the data.

    Accepts mlab quiver3d keyword arguments as keywords,
    which it will pass down.

    """
    assert isinstance(f, np.ndarray)

    from mayavi import mlab
    from dolfin.cpp import Mesh as dolfin_mesh
    from finmag.util.oommf.mesh import Mesh as oommf_mesh

    if isinstance(mesh, dolfin_mesh):
        coords = mesh.coordinates()
    elif isinstance(mesh, oommf_mesh):
        coords = np.array(list(mesh.iter_coords()))
    elif isinstance(mesh, np.ndarray) or isinstance(mesh, list):
        # If you know what the data has to look like, why not
        # be able to pass it in directly.
        coords = mesh
    else:
        raise TypeError("Don't know what to do with mesh of class {0}.".format(
            mesh.__class__))

    r = coords.reshape(coords.size, order="F").reshape((coords.shape[1], -1))
    # All 3 coordinates of the mesh points must be known to the plotter,
    # even if the mesh is one-dimensional. If not all coordinates are known,
    # fill the rest up with zeros.
    codimension = 3 - r.shape[0]
    if codimension > 0:
        r = np.append(r, [np.zeros(r[0].shape[0])] * codimension, axis=0)

    if f.size == f.shape[0]:
        # dolfin provides a flat numpy array, but we would like
        # one with the x, y and z components as individual arrays.
        f = components(f)

    figure = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    q = mlab.quiver3d(*(tuple(r) + tuple(f)), figure=figure, **kwargs)
    q.scene.isometric_view()
    mlab.title(title)
    mlab.axes(figure=figure)

    if filename:
        mlab.savefig(filename)
    else:
        mlab.show()
    mlab.close(all=True)


def plot_ndt_columns(ndt_file, columns=['m_x', 'm_y', 'm_z'], style='-',
                     xlim=None, ylim=None, outfile=None, title=None,
                     show_legend=True, legend_loc='best', figsize=None):
    """
    Helper function to quickly plot the time evolution of the specified
    columns in an .ndt file (default: m_x, m_y, m_z) and optionally save
    the output to a file.

    *Arguments*

    columns :  list of strings

        The names of the columns to plot. These must coincide with the
        names in the first header line of the .ndt file.

    outfile :  None | string

        If given, save the plot to a file with this name.

    style :  string

        The plotting style. Default: '-'.

    title :  None | string

        Title text to use for the plot.

    show_legend :  boolean

        If True, a legend with the column names is displayed.

    legend_loc :  string | integer

        Optional location code for the legend (same as for pyplot's
        legend() command).

    figsize :  None | pair of float

        Optional argument to set the figure size of the output plot.
    """
    f = Tablereader(ndt_file)
    ts = f.timesteps()
    column_vals = f[tuple(columns)]
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    for col_name, col in zip(columns, column_vals):
        ax.plot(ts, col, style, label=col_name)
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc=legend_loc)

    ymin, ymax = ylim or ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ymin - 0.05, ymax + 0.05)

    if outfile:
        fig.savefig(outfile)
    return fig


def plot_dynamics(ndt_file, components='xyz', **kwargs):
    """
    Convenience wrapper around `plot_ndt_columns` with a slightly
    easier to remember name. The main difference is that this function
    can only plot the dynamics of the magnetisation, not other fields.

    The magnetisation components to plot are specified in the argument
    `components`. All other keyword arguments are the same as for
    `plot_ndt_columns`.

    """
    columns = ['m_{}'.format(c) for c in components]
    return plot_ndt_columns(ndt_file, columns=columns, **kwargs)


def plot_dynamics_3d(ndt_file, field='m', style='-', elev=None, azim=None,
                     outfile=None, title=None, show_legend=True,
                     legend_loc='best', figsize=None):
    """
    Plot the time evolution of a 3D vector field (default: 'm') as a 3D
    trajectory and optionally save the output to a file.

    *Arguments*

    field :  string

        The field to be plotted (default: 'm'). The .ndt file must
        contain three columns corresponding to the x, y, z components
        of this field (e.g., 'm_x', 'm_y', 'm_z').

    style :  string

        The plotting style. Default: '-'.

    elev : float

        The 'elevation' (= polar angle) of the camera position (in degrees).
        Sensible values are between -90 and +90.

    azim : float

        The azimuthal angle of the camera position (in degrees).
        Sensible values are between 0 and 360.

    outfile :  None | string

        If given, save the plot to a file with this name.

    title :  None | string

        Title text to use for the plot.

    show_legend :  boolean

        If True, a legend with the field names is displayed.

    legend_loc :  string | integer

        Optional location code for the legend (same as for pyplot's
        legend() command).

    figsize :  None | pair of float

        Optional argument to set the figure size of the output plot.
    """
    f = Tablereader(ndt_file)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot('111', projection='3d')
    ax.view_init(elev=elev, azim=azim)
    fld_x = f[field + '_x']
    fld_y = f[field + '_y']
    fld_z = f[field + '_z']
    ax.plot(fld_x, fld_y, fld_z, label=field if show_legend else '')
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc=legend_loc)

    if outfile:
        fig.savefig(outfile)
    return fig


def plot_hysteresis_loop(H_vals, m_vals,
                         style='x-', add_point_labels=False, point_labels=None,
                         infobox=[], infobox_loc='bottom right',
                         filename=None, title="Hysteresis loop",
                         xlabel="H_ext (A/m)", ylabel="m_avg",
                         figsize=(10, 7)):
    """
    Produce a hysteresis plot.

    Arguments:

       H_vals -- list of scalar values; the values of the applied field used for the relaxation
                 stages of the hysteresis loop

       m_vals -- list of scalar values; the magnetisation obtained at the end of each relaxation
                 stage in the hysteresis loop

    Keyword arguments:

       style -- the plot style (default: 'x-')

       add_point_labels -- if True (default: False), every point is labeled with a number which
                           indicates the relaxation stage of the hysteresis loop it represents

       point_labels -- list or None; each list element can either be an integer or a pair of the
                       form (index, label), where index is an integer and label is a string. If
                       not None, only the points whose index appears in this list are labelled
                       (either with their index, or with the given label string if provided).
                       For example, if only every 10th point should be labeled, one might say
                       'point_labels=xrange(0, NUM_POINTS, 10)'.

       infobox -- list; each entry can be either a string or a pair of the form (name, value).
                  If not empty, an info box will added to the plot with the list elements appearing
                  on separate lines. Strings are printed verbatim, whereas name/value pairs are
                  converted to a string of the form "name = value".

       filename -- if given, save the resulting plot to a file with the specified name;
                   can also be a list of files

    """
    if not all([isinstance(x, (types.IntType, types.FloatType)) for x in m_vals]):
        raise ValueError(
            "m_vals must be a list of scalar values, got: {}".format(m_vals))

    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    N = len(H_vals) // 2
    H_max = max(H_vals)

    ax.plot(H_vals, m_vals, style)

    ax.set_xlim(-1.1 * H_max, 1.1 * H_max)
    ax.set_ylim((-1.2, 1.2))

    if point_labels is None:
        point_labels = xrange(len(H_vals))
    # Convert point_labels into a dictionary where the keys are the point indices
    # and the values are the respective labels to be used.
    point_labels = dict(
        map(lambda i: i if isinstance(i, tuple) else (i, str(i)), point_labels))
    if add_point_labels:
        for i in xrange(len(H_vals)):
            if point_labels.has_key(i):
                x = H_vals[i]
                y = m_vals[i]
                ax.annotate(point_labels[i], xy=(
                    x, y), xytext=(-10, 5) if i < N else (0, -15), textcoords='offset points')

    # draw the info box
    if infobox != []:
        box_text = ""
        for elt in infobox:
            if isinstance(elt, types.StringType):
                box_text += elt + '\n'
            else:
                try:
                    name, value = elt
                    box_text += "{} = {}\n".format(name, value)
                except ValueError:
                    raise ValueError(
                        "All list elements in 'infobox' must be either strings or pairs of the form (name, value). Got: '{}'".format(elt))
        box_text = box_text.rstrip()

        if infobox_loc not in ["top left", "top right", "bottom left", "bottom right"]:
            raise ValueError(
                "'infobox_loc' must be one of 'top left', 'top right', 'bottom left', 'bottom right'.")

        vpos, hpos = infobox_loc.split()
        x = H_max if hpos == "right" else -H_max
        y = 1.0 if vpos == "top" else -1.0

        ax.text(x, y, box_text, size=12,
                # transform = ax.transAxes,
                horizontalalignment=hpos, verticalalignment=vpos, multialignment="left",
                bbox=dict(boxstyle="round, pad=0.3", facecolor="white", edgecolor="green", linewidth=1))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()

    if filename:
        filenames = [filename] if isinstance(
            filename, basestring) else filename

        for name in filenames:
            create_missing_directory_components(name)
            fig.savefig(name)


def boxplot(arr, filename, **kwargs):
    plt.boxplot(list(arr), **kwargs)
    plt.savefig(filename)

if __name__ == "__main__":
    # --------------------------- DEMO ----------------------------------------#
    xs = np.linspace(-300, 300, 201)
    ts = np.linspace(0, 100, 101)

    my = np.zeros((len(ts), len(xs)))
    for t in ts:  # fake some magnetisation data
        my[t][:] = t * np.sin(
            2 * np.pi * 3 * xs / abs(np.min(xs) - np.max(xs))) / 100

    print("# values on x-axis: {}, # values on y-axis (time): {}.").format(
        len(xs), len(ts))
    print("Shape of the plotted array: {}.").format(my.shape)
    print("Minimum: {}, Maximum: {}.").format(np.min(my), np.max(my))

    labels = ("x (nm)", "time (ps)", "m_y")
    surface_2d(xs, ts, my, labels, "2D surface", path="surface_2d.png")
    surface_3d(xs, ts, my, labels, "3D surface", path="surface_3d.png")

    print("Saved plots in 'surface_2d.png' and 'surface_3d.png'.")
