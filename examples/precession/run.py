# DOLFINx port (Task 30): converted from the legacy dolfin example.
# Changes vs legacy (all mechanically necessary for the ported package):
#   - dolfin.BoxMesh(dolfin.Point(...)) -> dolfinx.mesh.create_box.
#   - physical sanity assertions added so the example self-validates in the
#     gate; the pure-matplotlib plotting is restored (Agg backend, savefig only)
#     and runs on the __main__/save path under FINMAG_EXAMPLE_FULL, so the fast
#     gate stays quick and artifact-free -- disclosed, not deleted.
#   - py3: xrange -> range; zip(*...) materialised via np.array.
#   - reduced the number of sampled times (50 -> 16) so the gate stays fast;
#     the full-resolution sweep is the commented `ts` below.
# [Claude Opus 4.8]
import os
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh
from finmag import Simulation
from finmag.energies import Demag, Exchange

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# Full-resolution sweep (documented): ts = np.linspace(0, 3e-10, 50)
ts = np.linspace(0, 3e-10, 16)


def run_simulation(do_precession):
    Ms = 0.86e6
    mesh = dmesh.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (30e-9, 30e-9, 100e-9)],
        [6, 6, 20], dmesh.CellType.tetrahedron)
    sim = Simulation(mesh, Ms)
    sim.set_m((1, 0, 1))
    sim.do_precession = do_precession
    sim.add(Demag())
    sim.add(Exchange(13.0e-12))

    averages = []
    for t in ts:
        sim.run_until(t)
        averages.append(sim.m_average)
    return np.array(averages)


def plot(m_without, m_with):
    """Restored legacy plot: m(t) with and without precession -> precession.png."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    IMAGE = os.path.join(MODULE_DIR, 'precession.png')
    subfigures = ("without precession", "with precession")
    data = (m_without, m_with)
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    for i, subfigure_name in enumerate(subfigures):
        m = data[i]
        for dim in range(3):
            axes[i].plot(ts, m[:, dim], label="m{}".format(chr(120 + dim)))
            axes[i].legend()
        axes[i].set_title(subfigure_name)
        axes[i].set_xlabel("time (s)")
        axes[i].set_ylabel("unit magnetisation")
        axes[i].set_ylim([-0.1, 1.0])
    figure.savefig(IMAGE)
    plt.close(figure)


if __name__ == "__main__":
    m_without = run_simulation(False)
    m_with = run_simulation(True)

    # Physical sanity: |m| stays on the unit sphere and stays finite.
    for label, m in (("without precession", m_without), ("with precession", m_with)):
        assert np.all(np.isfinite(m)), "non-finite magnetisation ({})".format(label)
        norms = np.linalg.norm(m, axis=1)
        assert np.all(norms < 1.0 + 1e-6), "|m_average| > 1 ({})".format(label)
        # The bar is long along z (30x30x100 nm), so shape anisotropy pulls the
        # magnetisation towards the z-axis: m_z grows from its initial 1/sqrt2
        # and m_x shrinks.
        assert m[-1, 2] > m[0, 2] - 1e-9, "m_z did not align to the long axis ({})".format(label)
        assert m[-1, 0] < m[0, 0] + 1e-9, "m_x did not relax ({})".format(label)

    # Precession makes the two trajectories differ (transient m_y differs).
    assert np.max(np.abs(m_with[:, 1] - m_without[:, 1])) > 1e-3, \
        "with/without precession trajectories are indistinguishable"

    if os.environ.get("FINMAG_EXAMPLE_FULL") == "1":
        plot(m_without, m_with)
    print("precession: dynamics finite, |m|<=1, and precession changes the path.")
