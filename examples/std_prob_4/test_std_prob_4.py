import os
import py
import numpy as np
import dolfin as df
import matplotlib.pyplot as plt
from math import sqrt
from finmag.util.meshes import from_geofile
from finmag.util.consts import mu0
from finmag import Simulation
from finmag.energies import Zeeman, Demag, Exchange
from finmag.util.timings import default_timer

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
m_0_file = os.path.join(MODULE_DIR, "m_0.npy")
m_at_crossing_file = os.path.join(MODULE_DIR, "m_at_crossing.npy")

"""
Micromag Standard Problem #4

specification:
    http://www.ctcms.nist.gov/~rdm/mumag.org.html

"""

Ms = 8.0e5
A = 1.3e-11
alpha = 0.02
gamma = 2.211e5


def create_initial_s_state():
    """
    Creates equilibrium s-state by slowly switching off a saturating field.

    """
    mesh = from_geofile(os.path.join(MODULE_DIR, "bar.geo"))

    sim = Simulation(mesh, Ms, name="relaxation", unit_length=1e-9)
    sim.alpha = 0.5  # good enough for relaxation
    sim.gamma = gamma
    sim.set_m((1, 1, 1))
    sim.add(Demag())
    sim.add(Exchange(A))

    # Saturating field of Ms in the [1, 1, 1] direction, that gets reduced
    # every 10 picoseconds until it vanishes after one nanosecond.
    H_initial = Ms * np.array((1, 1, 1)) / sqrt(3)
    H_multipliers = list(np.linspace(0, 1))
    H = Zeeman(H_initial)

    def lower_H(sim):
        try:
            H_mult = H_multipliers.pop()
            print "At t = {} s, lower external field to {} times initial field.".format(
                sim.t, H_mult)
            H.set_value(H_mult * H_initial)
        except IndexError:
            sim.llg.effective_field.interactions.remove(H)
            print "External field is off."
            return True

    sim.add(H)
    sim.schedule(lower_H, every=10e-12)
    sim.run_until(0.5e-9)
    sim.relax()

    np.save(m_0_file, sim.m)
    print "Saved magnetisation to {}.".format(m_0_file)
    print "Average magnetisation is ({:.2g}, {:.2g}, {:.2g}).".format(*sim.m_average)


def run_simulation(stop_when_mx_eq_zero):
    """
    Runs the simulation using field #1 from the problem description.

    Stores the average magnetisation components regularly, as well as the
    magnetisation when the x-component of the average magnetisation crosses
    the value 0 for the first time.

    """
    mesh = from_geofile(os.path.join(MODULE_DIR, "bar.geo"))

    sim = Simulation(mesh, Ms, name="dynamics", unit_length=1e-9)
    sim.alpha = alpha
    sim.gamma = gamma
    sim.set_m(np.load(m_0_file))
    sim.add(Demag())
    sim.add(Exchange(A))

    """
    Conversion between mu0 * H in mT and H in A/m.

                   mu0 * H = 1               mT
                           = 1 * 1e-3        T
                           = 1 * 1e-3        Vs/m^2
    divide by mu0 with mu0 = 4pi * 1e-7      Vs/Am
    gives                H = 1 / 4pi * 1e4   A/m

    with the unit A/m which is indeed what we want.
    Consequence:
        Just divide the value of mu0 * H in Tesla
        by mu0 to get the value of H in A/m.

    """
    Hx = -24.6e-3 / mu0
    Hy = 4.3e-3 / mu0
    Hz = 0
    sim.add(Zeeman((Hx, Hy, Hz)))

    def check_if_crossed(sim):
        mx, _, _ = sim.m_average
        print "The x-component of the averaged magnetisation is mx = {}.".format(mx)
        if mx <= 0:
            print "The x-component of the spatially averaged magnetisation first crossed zero at t = {}.".format(sim.t)
            np.save(m_at_crossing_file, sim.m)
            # When this function returns True, it means this event is done
            # and doesn't need to be triggered anymore.
            # When we return False, it means we want to stop the simulation.
            return not stop_when_mx_eq_zero

    sim.schedule(check_if_crossed, every=1e-12)
    sim.schedule(Simulation.save_averages, every=10e-12, at_end=True)
    sim.schedule(Simulation.save_vtk, every=10e-12, at_end=True)
    sim.run_until(2.0e-9)
    return sim.t


@py.test.mark.slow
def test_std_prob_4_field_1(stop_when_mx_eq_zero=True):
    REL_TOL = 0.01

    if not os.path.exists(m_0_file):
        print "Couldn't find initial magnetisation, creating one."
        create_initial_s_state()

    print "Running simulation..."
    t_0 = run_simulation(stop_when_mx_eq_zero) * 1e9
    print default_timer

    t_ref_martinez = 0.13949  # http://www.ctcms.nist.gov/~rdm/std4/Torres.html
    assert abs(t_0 - t_ref_martinez) / t_ref_martinez < REL_TOL


if __name__ == "__main__":
    test_std_prob_4_field_1(stop_when_mx_eq_zero=False)
