# DOLFINx port (Task 30): converted from the legacy dolfin example.
#
# Exponentially increasing damping at the film edges. The legacy script built a
# dolfin Expression, projected it onto a CG1 space, and plotted it. In the port:
#   - the damping profile is a plain vectorized NumPy callable (the port drops
#     string Expressions -- see INTERFACE-DRIFT: Expression-strings);
#   - df.project / matplotlib plotting are removed (plotting deferred, Task 26);
#   - from_geofile("film.geo") is kept UNCHANGED (Netgen-CSG loader ported for
#     the examples subset, Task 30 amendment -- film.geo exercises the
#     cylinder + capping-plane + 'and not' + multi-tlo path);
#   - the profile is validated numerically instead of plotted.
# [Claude Opus 4.8]
import os
import numpy as np
from finmag.util.meshes import from_geofile

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def damping_profile(alpha, xmin, xmax, width):
    """Return a vectorized callable alpha(x[0]) that ramps from `alpha` in the
    bulk up to ~1 within `width` of either x-edge (replacing the legacy
    df.Expression)."""
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


if __name__ == "__main__":
    mesh = from_geofile(os.path.join(MODULE_DIR, "film.geo"))
    xs = mesh.geometry.x[:, 0]
    xmin, xmax = xs.min(), xs.max()
    f = damping_profile(0.02, 0, 1000, 200)
    alphas = f(np.linspace(xmin, xmax, 1000))

    # The bulk damping is ~0.02, it rises towards 1 near the edges, and never
    # exceeds 1.
    assert np.isclose(alphas.min(), 0.02, atol=1e-6), "bulk damping wrong"
    assert alphas.max() > 0.9, "edge damping did not ramp up"
    assert alphas.max() <= 1.0 + 1e-6, "damping exceeded 1"
    # film.geo (x in [0, 1000]) loaded through the ported Netgen-CSG reader.
    assert xmin < 1.0 and xmax > 999.0, "unexpected film x-extent"
    print("edge_damping: film.geo loaded; edge-damping profile is correct.")
