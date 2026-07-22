# DOLFINx port (Task 30): converted from the legacy dolfin example.
#
# Exponentially increasing damping at the film edges. The legacy script built a
# dolfin Expression, projected it onto a CG1 space, and plotted it. In the port:
#   - the damping profile is a plain vectorized NumPy callable (the port drops
#     string Expressions -- see INTERFACE-DRIFT: Expression-strings). The legacy
#     function NAME `damping_expression` is kept unchanged.
#   - df.project is removed (the callable is evaluated directly), but the
#     pure-matplotlib plot_damping_profile is restored (Agg backend, savefig
#     only) and runs on the __main__/save path under FINMAG_EXAMPLE_FULL, so the
#     fast gate stays quick and artifact-free -- disclosed, not deleted.
#   - from_geofile("film.geo") is kept UNCHANGED (Netgen-CSG loader ported for
#     the examples subset, Task 30 amendment -- film.geo exercises the
#     cylinder + capping-plane + 'and not' + multi-tlo path);
#   - the profile is validated numerically as well as plotted.
# [Claude Opus 4.8]
import os
import numpy as np
from finmag.util.meshes import from_geofile

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def damping_expression(alpha, xmin, xmax, width):
    """Return a vectorized callable alpha(x[0]) that ramps from `alpha` in the
    bulk up to ~1 within `width` of either x-edge (replacing the legacy
    df.Expression; the function name is kept from legacy)."""
    eps = 0.01
    a = alpha + eps
    b = ((1 + eps) / a) ** (1.0 / width)
    xa = xmin + width
    xb = xmax - width

    def f(xs):
        xs = np.asarray(xs, dtype=float)
        near_left = xs <= xa
        near_right = xb <= xs
        ref = np.where(near_left, xa, xb)
        edge_val = a * b ** np.abs(xs - ref) - eps
        return np.where(near_left | near_right, edge_val, alpha)

    return f


def plot_damping_profile(f, mesh):
    """Plot a given damping profile to file 'damping.png'.

    `f` is the vectorized callable returned by damping_expression; `mesh` is the
    DOLFINx mesh (its x-extent sets the plot range). Restored from legacy;
    replaces the legacy df.project + point-eval with a direct callable eval.
    """
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    xs = mesh.geometry.x[:, 0]
    xmin = xs.min()
    xmax = xs.max()

    points = 1000
    xs_plot = np.linspace(xmin, xmax, points)
    alphas = f(xs_plot)

    plt.plot(xs_plot, alphas)
    plt.xlabel("x (nm)")
    plt.xlim((xmin, xmax))
    plt.ylabel("damping")
    plt.ylim((0, 1))
    plt.grid()
    plt.title("Spatial Profile of the Damping")
    plt.savefig(os.path.join(MODULE_DIR, 'damping.png'))
    plt.close()
    print("Saved plot of damping to 'damping.png'.")


if __name__ == "__main__":
    mesh = from_geofile(os.path.join(MODULE_DIR, "film.geo"))
    xs = mesh.geometry.x[:, 0]
    xmin, xmax = xs.min(), xs.max()
    f = damping_expression(0.02, 0, 1000, 200)
    alphas = f(np.linspace(xmin, xmax, 1000))

    # The bulk damping is ~0.02, it rises towards 1 near the edges, and never
    # exceeds 1.
    assert np.isclose(alphas.min(), 0.02, atol=1e-6), "bulk damping wrong"
    assert alphas.max() > 0.9, "edge damping did not ramp up"
    assert alphas.max() <= 1.0 + 1e-6, "damping exceeded 1"
    # film.geo (x in [0, 1000]) loaded through the ported Netgen-CSG reader.
    assert xmin < 1.0 and xmax > 999.0, "unexpected film x-extent"

    if os.environ.get("FINMAG_EXAMPLE_FULL") == "1":
        plot_damping_profile(f, mesh)
    print("edge_damping: film.geo loaded; edge-damping profile is correct.")
