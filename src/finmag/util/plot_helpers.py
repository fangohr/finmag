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


def surface_2d(x, y, u, labels=("", "", ""), title="", ylim=None, xlim=None, clim=None, path=""):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    X, Y = np.meshgrid(x, y)
    surf = ax.pcolormesh(X, Y, u)

    ax.set_xlabel(labels[0])
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if clim is not None:
        surf.set_clim(vmin=clim[0], vmax=clim[1])
    ax.set_ylabel(labels[1])
    cb = fig.colorbar(surf)
    cb.ax.yaxis.set_label_position('right')
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

if __name__ == "__main__":
    # --------------------------- DEMO ----------------------------------------#
    xs = np.linspace(-300, 300, 201)
    ts = np.linspace(0, 100, 101)

    my = np.zeros((len(ts), len(xs)))
    for t in ts:  # fake some magnetisation data
        my[t][:] = t * np.sin(
            2 * np.pi * 3 * xs / abs(np.min(xs) - np.max(xs))) / 100

    print "# values on x-axis: {}, # values on y-axis (time): {}.".format(
        len(xs), len(ts))
    print "Shape of the plotted array: {}.".format(my.shape)
    print "Minimum: {}, Maximum: {}.".format(np.min(my), np.max(my))

    labels = ("x (nm)", "time (ps)", "m_y")
    surface_2d(xs, ts, my, labels, "2D surface", "surface_2d.png")
    surface_3d(xs, ts, my, labels, "3D surface", "surface_3d.png")

    print "Saved plots in 'surface_2d.png' and 'surface_3d.png'."
