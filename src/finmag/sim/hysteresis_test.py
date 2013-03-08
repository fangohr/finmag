import dolfin as df
import numpy as np
from finmag import sim_with
from finmag.util.helpers import plot_hysteresis_loop

ONE_DEGREE_PER_NS = 17453292.5 # in rad/s

def test_hysteresis_loop_and_plotting():
    mesh = df.BoxMesh(0, 0, 0, 1, 1, 1, 1, 1, 1)
    sim = sim_with(mesh, Ms=1e6, m_init=(0.8, 0.2, 0), alpha=1.0, unit_length=1e-9, A=None, demag_solver=None)#'FK')

    H = 0.2e6  # maximum external field strength in A/m
    initial_direction = np.array([1.0, 0.01, 0.0])
    N = 5

    (H_vals, m_vals) = sim.hysteresis_loop(H, initial_direction, N, stopping_dmdt=10)

    assert(np.allclose(m_vals, [1.0 for _ in xrange(2*N)], atol=1e-4))

    # This only tests whether the plotting function works without
    # errors. It currently does *not* check that it produces
    # meaningful results (and the plot is quite boring for the system
    # above anyway).
    plot_hysteresis_loop(H_vals, m_vals, infobox=["param_A = 23", ("param_B", 42)], title="Hysteresis plot test",
                         xlabel="H_ext", ylabel="m_avg", figsize=(5,4), infobox_loc="bottom left")
