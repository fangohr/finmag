import os, sys
import dolfin as df
import numpy as np
from finmag import Simulation
from finmag.energies import Zeeman, TimeZeeman, Demag, Exchange

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

"""
Micromag Standard Problem #4

specification:
    http://www.ctcms.nist.gov/~rdm/mumag.org.html

"""

L = 500e-9; W = 125e-9; H = 3e-9; # dimensions of film          m
Ms = 8.0e5                        # saturation magnetisation    A/m
A = 1.3e-11                       # exchange coupling strength  J/m
alpha = 0.02
gamma = 2.211e5                   # in m/(As)

mesh = df.Box(0, 0, 0, L, W, H, 166, 41, 1)

sim = Simulation(mesh, Ms)
sim.alpha = alpha
sim.gamma = gamma
sim.set_m((1, 1, 1))
sim.add(Demag())
sim.add(Exchange(A))

def initialise_m():
    sim.alpha = 1
    sim.llg.do_precession = False

    # time dependent field, which fades until one nanosecond
    # and gets updated every 10 picoseconds.
    t_init = 0; t_max = 1e-9; t_update = 1e-11;
    fx = fy = fz = "(t_max - t) * H"
    f_expr = df.Expression((fx, fy, fz), t=t_init, t_max=t_max, H=Ms)
    saturating_field = TimeZeeman(f_expr, np.arange(t_init, t_max, t_update))
    sim.add(saturating_field)
   
    def update_saturating_field(llg):
        if not saturating_field.switched_off:
            saturating_field.update(llg.t)
    sim.llg._pre_rhs_callables.append(update_saturating_field)

    sim.run_until(1.5e-9)
    np.savetxt("m_init.txt", sim.m)
    print "Saved magnetisation to m_init.txt."

if not os.path.exists(MODULE_DIR + "/m_init.txt"):
    initialise_m()
    print "Magnetisation saved, please restart script."
    sys.exit()
else:
    m_values = np.loadtxt(MODULE_DIR + "/m_init.txt")

sim.set_m(m_values)
# df.plot(sim.llg._m) # looks like an s-state alright
# df.interactive()

# field 1
# mu0 * Hx = -24,6 mT = -24,6 * 10^-3 Vs/m^2 divide by mu0
# Hx = -24,6 / 4PI * 10^-3 * 10*7 Am/Vs * Vs/m^2
#    = -24,6 / 4PI * 10^4 A/m
# Hy = 4,3 / 4PI * 10^4 A/m
# Hz = 0

Hx = -24.6e4 / (4 * np.pi)
Hy = 4.3e4 / (4 * np.pi)
Hz = 0
sim.add(Zeeman((Hx, Hy, Hz)))

f = open(MODULE_DIR + "/m_averages.txt", "w")
t = 0; t_max = 2e-9; dt = 1e-11;
while t <= t_max:
    sim.run_until(t_max)
    mx, my, mz = sim.m_average
    f.write("{} {} {} {}\n".format(t, mx, my, mz))
    t += dt
f.close()
