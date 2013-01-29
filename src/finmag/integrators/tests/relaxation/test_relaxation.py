import os
import numpy as np
import dolfin as df
import matplotlib.pyplot as plt
from finmag import Simulation
from finmag.energies import Zeeman, Exchange, Demag
from finmag.integrators.common import ONE_DEGREE_PER_NS

MODULE_DIR = os.path.abspath(os.path.dirname(__file__))

def test_easy_relaxation(do_plot=False):
    """
    This is a simulation we expect to relax well, meant to catch some obvious
    errors in the relaxation code.

    """
    mesh = df.Box(0, 0, 0, 50, 10, 10, 10, 2, 2)
    Ms = 0.86e6
    A = 13.0e-12

    sim = Simulation(mesh, Ms, name="test_relaxation")
    sim.set_m((1, 0, 0))
    sim.add(Zeeman((0, Ms, 0)))
    sim.add(Exchange(A))
    sim.add(Demag())
    sim.schedule(Simulation.save_averages, every=1e-12, at_end=True)
    sim.relax()

    if do_plot:
        plot_averages(sim)

    assert sim.t < 3e-10

def plot_averages(sim):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    t, mx, my, mz = np.array(zip(* np.loadtxt("test_relaxation.ndt")))
    t = t * 1e9
    ax1.plot(t, mx, "b-", label="$m_\mathrm{x}$")
    ax1.plot(t, my, "b--", label="$m_\mathrm{y}$")
    ax1.plot(t, mz, "b:", label="$m_\mathrm{z}$")
    ax1.set_xlabel("time (ns)")
    ax1.set_ylabel("average magnetisation", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")
    ax1.legend()

    ax2 = ax1.twinx()
    t, max_dmdt_norms = np.array(zip(* sim.integrator.all_max_dmdt_norms))
    ax2.semilogy(t*1e9, max_dmdt_norms/ONE_DEGREE_PER_NS, "ro")
    ax2.set_ylabel("maximum dm/dt (1/ns)", color="r")
    ax2.axhline(y=1, xmin=0.5, color="red", linestyle="--")
    ax2.annotate("threshold", xy=(0.15, 1.1), color="r")
    #ax2.set_ylabel(r"$\max \left( \left | \frac{\mathrm{d}m}{\mathrm{d}t} \right | \right) \, \left( \mathrm{ns}^{-1} \right )$", color="r", fontsize=16)
    for tl in ax2.get_yticklabels():
        tl.set_color("r")
    plt.savefig(os.path.join(MODULE_DIR, "m_and_dmdt.png"))

if __name__ == "__main__":
   test_easy_relaxation(do_plot=True) 
