import os
import dolfin as df
import numpy as np
from progressbar import ProgressBar, Percentage, Bar, ETA
from finmag.util.convert_mesh import convert_mesh
from finmag import Simulation
from finmag.energies import Zeeman, DiscreteTimeZeeman, Demag, Exchange

MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"
mesh_file = MODULE_DIR + "bar.geo"
initial_m_file = MODULE_DIR + "m_init.txt"
average_m_file = MODULE_DIR + "m_averages.txt"
zero_crossing_m_file = MODULE_DIR + "m_zero.txt"

"""
Micromag Standard Problem #4

specification:
    http://www.ctcms.nist.gov/~rdm/mumag.org.html

"""

Ms = 8.0e5; A = 1.3e-11; alpha = 0.02; gamma = 2.211e5
mesh = df.Mesh(convert_mesh(mesh_file))

def create_initial_s_state():
    """
    Create equilibrium s-state by slowly switching off a saturating field.

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
    print "Average magnetisation is ({:.2}, {:.2}, {:.2}).".format(*sim.m_average)

def run_simulation(m_file):
    sim = Simulation(mesh, Ms, unit_length=1e-9)
    sim.alpha = alpha
    sim.gamma = gamma
    sim.set_m(np.loadtxt(m_file))
    sim.add(Demag())
    sim.add(Exchange(A))

    # Field 1
    # mu_0 * H_x = 24.6 mT, so in SI units: H_x = 24.6 * 10^4 / 4 PI A/m
    Hx = -24.6e4 / (4 * np.pi)
    Hy = 2 * 4.3e4 / (4 * np.pi)
    Hz = 0
    sim.add(Zeeman((Hx, Hy, Hz)))

    t = 0; t_max = 1.1e-10; dt = 1e-14;
    mx_crossed_zero = False
    with open(average_m_file, "w") as f:
        widgets = [Percentage(), " ", Bar(), " ", ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=t_max+dt)
        pbar.start()
        while t <= t_max:
            sim.run_until(t)

            # Write average magnetisation to file.
            mx, my, mz = sim.m_average
            f.write("{} {} {} {}\n".format(t, mx, my, mz))

            # Store magnetisation at moment of m_x crossing zero as well.
            if mx <= 0 and not mx_crossed_zero:
                print "m_x crossed 0 at {}.".format(t)
                np.savetxt(zero_crossing_m_file, sim.m)
                mx_crossed_zero = True

            t += dt
            pbar.update(t)
        pbar.finish()

if __name__ == "__main__":
    if not os.path.exists(initial_m_file):
        print "Couldn't find initial magnetisation, creating one."
        create_initial_s_state()
    print "Running simulation..."
    run_simulation(initial_m_file)
