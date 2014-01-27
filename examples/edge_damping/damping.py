import numpy as np
import dolfin as df
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def damping_expression(alpha, xmin, xmax, width):
    """
    Exponentially increasing damping at the edges along the x-axis.

    Will increase alpha from its starting value `alpha` to 1 in a region
    less than `width` away from `xmin` or `xmax`, where `xmin` and `xmax` are
    the extremal x-coordinates of the mesh. Returns a df.Expression.

    """
    eps = 0.01  # changes slope of exponential fit
    a = alpha + eps
    b = ((1 + eps) / a) ** (1.0 / width)
    xa = xmin + width
    xb = xmax - width
    code = ("(x[0] <= xa || xb <= x[0])"
            " ? a * pow(b, fabs(x[0] - (x[0] <= xa ? xa : xb))) - eps"
            " : alpha")
    expr = df.Expression(code, xa=xa, xb=xb, alpha=alpha, a=a, b=b, eps=eps)
    return expr


def plot_damping_profile(expr, mesh):
    """
    Plot a given damping profile to file 'damping.png'.

    The first argument `expr` should be a df.Expression (it can be obtained
    using the function damping_expression in this module) and the second
    argument should be a df.Mesh.

    """
    xs = mesh.coordinates()[:, 0]
    xmin = xs.min()
    xmax = xs.max()

    points = 1000
    xs_plot = np.linspace(xmin, xmax, points)
    alphas = np.zeros(points)

    S1 = df.FunctionSpace(mesh, "CG", 1)
    alpha_func = df.project(expr, S1)
    for i, x in enumerate(xs_plot):
        try:
            alphas[i] = alpha_func(x, 0, 0)
        except RuntimeError:
            # could raise Exception due to now resolved bug in dolfin
            # https://bitbucket.org/fenics-project/dolfin/issue/97/function-eval-does-not-find-a-point-that
            alphas[i] = 0

    plt.plot(xs_plot, alphas)
    plt.xlabel("x (nm)")
    plt.xlim((xmin, xmax))
    plt.ylabel("damping")
    plt.ylim((0, 1))
    plt.grid()
    plt.title("Spatial Profile of the Damping")
    plt.savefig('damping.png')
    print "Saved plot of damping to 'damping.png'."

if __name__ == "__main__":
    from finmag.util.meshes import from_geofile
    mesh = from_geofile("film.geo")
    expr = damping_expression(0.02, 0, 1000, 200)
    plot_damping_profile(expr, mesh)
