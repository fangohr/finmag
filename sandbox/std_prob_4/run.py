import os
import dolfin as df
import numpy as np
from finmag.util.meshes import from_geofile
from finmag import Simulation
from finmag.energies import Zeeman, DiscreteTimeZeeman, Demag, Exchange

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
mesh_file = os.path.join(MODULE_DIR, "bar.geo")
initial_m_file = os.path.join(MODULE_DIR, "m_init.txt")
average_m_file = os.path.join(MODULE_DIR, "m_averages.txt")
zero_crossing_m_file = os.path.join(MODULE_DIR, "m_zero.txt")

"""
Micromag Standard Problem #4

specification:
    http://www.ctcms.nist.gov/~rdm/mumag.org.html

"""

Ms = 8.0e5; A = 1.3e-11; alpha = 0.02; gamma = 2.211e5
mesh = from_geofile(mesh_file)

def create_initial_s_state():
    """
    Creates equilibrium s-state by slowly switching off a saturating field.

    """
    sim = Simulation(mesh, Ms, unit_length=1e-9)
    sim.gamma = gamma
    sim.set_m((1, 1, 1))
    sim.add(Demag())
    sim.add(Exchange(A))

    # We're not interested in the dynamics, this will speed up the simulation.
    sim.alpha = 1
    sim.llg.do_precession = False

    # Saturating field in the [1, 1, 1] direction, that gets reduced
    # every 10 picoseconds until it vanishes after one nanosecond.
    t_off = 1e-9; dt_update = 1e-11;
    fx = fy = fz = "(1 - t/t_off) * H"
    f_expr = df.Expression((fx, fy, fz), t=0.0, t_off=t_off, H=Ms)
    saturating_field = DiscreteTimeZeeman(f_expr, t_off, dt_update)
    sim.add(saturating_field, with_time_update=saturating_field.update)

    sim.run_until(2e-9)
    np.savetxt(initial_m_file, sim.m)
    print "Saved magnetisation to {}.".format(initial_m_file)
    print "Average magnetisation is ({:.2g}, {:.2g}, {:.2g}).".format(*sim.m_average)

def run_simulation():
    """
    Runs the simulation using field #1 from the problem description.

    Stores the average magnetisation components regularly, as well as the
    magnetisation when the x-component of the average magnetisation crosses
    the value 0 for the first time.

    """
    sim = Simulation(mesh, Ms, unit_length=1e-9)
    sim.alpha = alpha
    sim.gamma = gamma
    sim.set_m(np.loadtxt(initial_m_file))
    sim.add(Demag())
    sim.add(Exchange(A))

    # Field 1
    # mu_0 * H_x = 24.6 mT, so in SI units: H_x = 24.6 * 10^4 / 4 PI A/m
    Hx = -24.6e4 / (4 * np.pi)
    Hy = 2 * 4.3e4 / (4 * np.pi)
    Hz = 0
    sim.add(Zeeman((Hx, Hy, Hz)))

    t = 0; t_counter = 0; t_max = 2e-9; dt = 1e-14;
    mx_crossed_zero = False
    with open(average_m_file, "w") as f:
        while t <= t_max:
            sim.run_until(t)

            mx, my, mz = sim.m_average
            # Store magnetisation when m_x crosses zero.
            if mx <= 0 and not mx_crossed_zero:
                print "m_x crossed 0 at {}.".format(t)
                np.savetxt(zero_crossing_m_file, sim.m)
                mx_crossed_zero = True

            if t_counter % 1000 == 0:
                # Write average magnetisation to file every 1000 timesteps.
                f.write("{} {} {} {}\n".format(t, mx, my, mz))

            t += dt; t_counter += 1;

if __name__ == "__main__":
    if not os.path.exists(initial_m_file):
        print "Couldn't find initial magnetisation, creating one."
        create_initial_s_state()
    print "Running simulation..."
    run_simulation()
