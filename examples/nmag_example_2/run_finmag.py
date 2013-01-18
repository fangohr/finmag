import dolfin as df
from finmag import Simulation
from finmag.energies import Exchange, Demag
from finmag.util.timings import timings

def run_simulation():
    mesh = df.Mesh("bar.xml.gz")

    sim = Simulation(mesh, Ms=0.86e6, unit_length=1e-9, name="finmag_bar")
    sim.set_m((1, 0, 1))
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
