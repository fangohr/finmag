from finmag import Simulation
from finmag.integrators.llg_integrator import llg_integrator
from finmag.energies import Exchange, Demag
from finmag.util.timings import timings
import dolfin as df

def run_simulation():
    mesh = df.Mesh("bar.xml.gz")

    sim = Simulation(mesh, Ms=0.86e6, unit_length=1e-9, name="finmag_bar")
    sim.set_m((1, 0, 1))
    sim.integrator = llg_integrator(sim.llg, sim.llg.m, backend=sim.integrator_backend, reltol=1e-6, abstol=1e-6)
    sim.alpha = 0.5

    sim.add(Exchange(13.0e-12))
    sim.add(Demag())

    sim.schedule(Simulation.save_averages, every=5e-12)
    sim.run_until(3e-10, save_averages=False)

    print timings
    print "The RHS was evaluated {} times, while the Jacobian was computed {} times.".format(
            sim.integrator.stats()['nfevals'],
            timings._timings['LLG::sundials_jtimes'].calls)

if __name__ == "__main__":
    run_simulation()
