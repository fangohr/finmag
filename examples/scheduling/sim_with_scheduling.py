import logging
import dolfin as df
from finmag import Simulation
from finmag.energies import Exchange, Demag, Zeeman

log = logging.getLogger(name="finmag")
log.setLevel(logging.ERROR) # To better show output of this program.

# Three example functions that will be added to the schedule.
# They will get called with the simulation object as their first parameter.

def progress(s):
    print "We have integrated up to t = {:.3f} ns.".format(1e9 * s.t)
    print "Average magnetisation is m = {}.".format(s.m_average)

def halfway_done(s):
    print "We are halfway done!"

def done(s):
    print "Woohoo!"

def example_simulation():
    Ms = 8.6e5
    mesh = df.Box(0, 0, 0, 40, 20, 20, 10, 5, 5)

    example = Simulation(mesh, Ms, name="sim_with_scheduling")
    example.set_m((0.1, 1, 0))
    example.add(Exchange(13.0e-12))
    example.add(Demag())
    example.add(Zeeman((Ms/2, 0, 0)))

    return example


if __name__ == "__main__":

    # Assemble a simulation as usual.
    sim = example_simulation()

    # Add the functions to the scheduler using sim.schedule.

    t_final = 1.05e-9

    sim.schedule(progress, every=1e-10, at_end=True) # `every` keyword with optional `at_end`
    sim.schedule(halfway_done, at=t_final/2) # `at` keyword
    sim.schedule(done, at_end=True) # `at_end` used on its own

    # Now integrate. The functions will get called according to the schedule.

    sim.run_until(t_final)
