import os
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
from finmag import Simulation as Sim
from finmag.energies import Zeeman

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def test_sim_ode(do_plot=False):
    mesh = df.Box(0, 0, 0, 2, 2, 2, 1, 1, 1)
    sim = Sim(mesh, 8.6e5, unit_length=1e-9)
    sim.alpha = 0.1
    sim.set_m((1, 0, 0))

    H0 = 1e5
    sim.add(Zeeman((0, 0, H0)))

    dt = 1e-12; ts = np.linspace(0, 100 * dt, 100)

    precession_coeff = sim.gamma / (1 + sim.alpha ** 2)
    mz_ref = np.tanh(precession_coeff * sim.alpha * H0 * ts)

    mz = []
    for t in ts:
        sim.run_until(t, save_averages=False)
        mz.append(sim.m[-1]) # same as m_average for this macrospin problem

    if do_plot:
        ts_ns = ts * 1e9
        plt.plot(ts_ns, mz, "b.", label="computed") 
        plt.plot(ts_ns, mz_ref, "r-", label="analytical") 
        plt.xlabel("time (ns)")
        plt.ylabel("mz")
        plt.title("integrating a macrospin")
        plt.legend()
        plt.savefig(os.path.join(MODULE_DIR, "test_sim_ode.png"))

    assert np.max(np.abs(mz - mz_ref)) < 1.5e-8

if __name__ == "__main__":
    test_sim_ode(do_plot=True)
