import time
import numpy as np
import matplotlib.pyplot as plt
from finmag.util.meshes import cylinder
from finmag.util.consts import flux_density_to_field_strength
from finmag.util.helpers import spherical_to_cartesian
from finmag import Simulation
from finmag.energies import Exchange, Zeeman, UniaxialAnisotropy, Demag

Ms_Tesla = 0.5
Ms = flux_density_to_field_strength(Ms_Tesla)
A = 1e-11
K1 = 3e5
K1_axis = (0, 0, 1)
H_baseline = 2 * K1 / Ms_Tesla
H_mult = [0.95, 1.2, 2.8]
H_axis = - spherical_to_cartesian((1.0, 1.0 * np.pi/180, 0))
mesh = cylinder(6, 20, 2.0)

start_clock = time.clock()

for H in H_mult:
    sim = Simulation(mesh, Ms, unit_length=1e-9)
    sim.alpha = 0.02
    sim.set_m(K1_axis)
    sim.add(Exchange(A))
    sim.add(UniaxialAnisotropy(K1, K1_axis))
    sim.add(Demag())
    sim.add(Zeeman(H * H_baseline * H_axis))

    print "Running simulation for H = {} T.".format(H)
    print sim.mesh_info()

    t = 0; dt = 1e-12; t_failsafe = 2e-9;
    dt_output = 100e-12;

    ts = []; mzs = []
    while True:
        sim.run_until(t)
        m = sim.m_average
        ts.append(t*1e9); mzs.append(m[2])		

        if t % dt_output:
            print "at t = {:.2} ns, m = {}.".format(1e9*t, m)
        if m[2] < -0.5 or t >= t_failsafe:
            break

        t += dt

    plt.plot(ts, mzs, label="$H_\mathrm{{ext}} = {}$".format(H))

stop_clock = time.clock() 
print 'Simulations ran in {} seconds'.format(round(stop_clock - start_clock))

plt.xlabel('$\mathrm{{time (ns)}}$')
plt.ylabel('$m_\mathrm{{z}} (M_\mathrm{{S}})$')
plt.ylim([-0.6, 1])
plt.title("$m_\mathrm{z}$ as a function of time")
plt.legend()
plt.savefig("mz.png")
