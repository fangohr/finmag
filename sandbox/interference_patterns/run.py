import dolfin as df
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from finmag import Simulation
from finmag.util.consts import mu0
from finmag.energies import Zeeman, Exchange, ThinFilmDemag
from point_contacts import point_contacts

def figure_5_6(mesh, m, p, t):
    print "Saving the plot..."
    coords = mesh.coordinates()
    upper_z_plane = coords[coords.shape[0]/2:,0:2]
    m = m.view().reshape((3, -1))
    upper_mx = m[0,m.shape[1]/2:]
    #plt.contourf(upper_z_plane[:,0], upper_z_plane[:,1], upper_mx)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #X, Y, Z = axes3d.get_test_data(0.05)
    #ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
    ax.plot_wireframe(upper_z_plane[:,0], upper_z_plane[:,1], upper_mx, rstride=10, cstride=10)

    plt.savefig("mx_p{}_t{:.3f}ns.png".format(p, t*1e9))
    print "Saved."

alpha = 0.012 # dimensionless
gamma = 2.210173e5 # m/(As), our value

# Permalloy.
Ms = 860e3 # A/m
A = 13.0e-12 # J/m, our value, paper uses D/(gamma*h_bar) without giving the number

# Mesh
# Use dolfin for debugging, use geo file later.
# The film described in figures 5 and 6, page 7 and section 5, page 8 is
# 440 nm * 440 nm * 5 nm. We'll put the center of the film at 0,0,0.

L = 440e-9
W = 440e-9
H = 5e-9
dL = L/2; dW = W/2; dH = H/2;

# Use exchange length to figure out discretisation.
l_ex = np.sqrt(2 * A / (mu0 * Ms**2))
print "The exchange length is l_ex = {:.2} m.".format(l_ex) # >5 nm

discretisation = math.floor(l_ex*1e9)*1e-9 
nx, ny, nz = (L/discretisation, W/discretisation, 1)
#mesh = df.Rectangle(-dL, -dW, dL, dW, int(nx), int(ny)
mesh = df.Box(-dL, -dW, -dH, dL, dW, dH, int(nx), int(ny), int(nz))

sim = Simulation(mesh, Ms)
sim.set_m((0, 0, 1))
sim.add(Zeeman((0, 0, 1.1 * Ms))) # section 3 and end of section 5
sim.add(Exchange(A))
sim.add(ThinFilmDemag())

# Spin-Torque
I = 10e-3 # A
point_contact_radius = 10e-9
point_contact_area = math.pi * point_contact_radius ** 2
J = I / point_contact_area
print "Current density is J = {:.2} A/m^2.".format(J)
J_expr = point_contacts([(L/3, W/2), (2*L/3, W/2)],
        radius=point_contact_radius, J=J)
P = 0.4
d = H
p = (1, 0, 0)
sim.set_stt(J_expr, P, d, p)
pulse_time = 50e-12
snapshot_times = np.array([15e-12, 175e-12, 265e-12])

t = 0; dt = 1e-12; t_max = 1e-9;
while t <= t_max:
    if abs(t - 50e-12) < 1e-16:
        print "Switching spin current off at {}.".format(t)
        sim.toggle_stt(False)
    if np.min(np.abs(t - snapshot_times)) < 1e-16:
        figure_5_6(mesh, sim.m, "100", t)
    t += dt
    sim.run_until(t)
