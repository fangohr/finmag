from aeon import timer
from finmag import Simulation
from finmag.energies import Exchange, Demag
from finmag.util.meshes import from_geofile


def run_simulation(verbose=False):
    mesh = from_geofile('bar.geo')

    sim = Simulation(mesh, Ms=0.86e6, unit_length=1e-9, name="finmag_bar")
    sim.set_m((1, 0, 1))
    sim.set_tol(1e-6, 1e-6)
    sim.alpha = 0.5

    sim.add(Exchange(13.0e-12))
    sim.add(Demag())

    sim.schedule('save_averages', every=5e-12)
    sim.schedule("eta", every=10e-12)
    sim.run_until(3e-10)

    print timer
    if verbose:
        print "The RHS was evaluated {} times, while the Jacobian was computed {} times.".format(
                sim.integrator.stats()['nfevals'],
                timer.get("sundials_jtimes", "LLG").calls)

if __name__ == "__main__":
    run_simulation(verbose=False)
