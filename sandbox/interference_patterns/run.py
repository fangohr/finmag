import dolfin as df
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from finmag import Simulation
from finmag.util.consts import mu0
from finmag.energies import Zeeman, Exchange, ThinFilmDemag
from point_contacts import point_contacts

epsilon = 1e-16

alpha = 0.012 # dimensionless

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

discretisation = math.floor(l_ex*1e9)*1e-9 / 2
nx, ny, nz = (L/discretisation, W/discretisation, 1)
#mesh = df.Rectangle(-dL, -dW, dL, dW, int(nx), int(ny)
mesh = df.Box(-dL, -dW, -dH, dL, dW, dH, int(nx), int(ny), int(nz))

sim = Simulation(mesh, Ms)
sim.alpha = alpha
sim.set_m((0, 0, 1))
sim.add(Zeeman((0, 0, 1.1 * Ms))) # section 3 and end of section 5
sim.add(Exchange(A))
sim.add(ThinFilmDemag())

# Spin-Torque
I = 10e-3 # A
point_contact_radius = 10e-9
point_contact_area = math.pi * point_contact_radius ** 2
distance = 125e-9
J = I / point_contact_area
print "Current density is J = {:.2} A/m^2.".format(J)
J_expr_fig5_two_pc = point_contacts([(-distance/2, 0), (distance/2, 0)],
        radius=point_contact_radius, J=J) 

distance_fig_6 = 65e-9
phi0 = 0; phi1 = 2*math.pi/3; phi2 = 4*math.pi/3; r = distance_fig_6 * math.sqrt(3)/3
x0, y0 = r * math.cos(phi0), r * math.sin(phi0)
x1, y1 = r * math.cos(phi1), r * math.sin(phi1)
x2, y2 = r * math.cos(phi2), r * math.sin(phi2)
J_expr_fig6_three_pc = point_contacts([(x0, y0),(x1, y1),(x2, y2)], radius=point_contact_radius, J=J)

#J_visu = df.interpolate(J_expr_fig6_three_pc, sim.S1)
#df.plot(J_visu)
#df.interactive()
P = 0.4
p = (1, 1, 0)
d = H
sim.set_stt(J_expr_fig6_three_pc, P, d, p)
pulse_time = 50e-12
snapshot_times = np.array([65e-12, 175e-12, 265e-12])

t = 0; dt = 1e-12; t_max = 1e-9;
while t <= t_max:
    if abs(t - pulse_time) < epsilon:
        print "Switching spin current off at {}.".format(t)
        sim.toggle_stt(False)
    if np.min(np.abs(t - snapshot_times)) < epsilon:
        sim.snapshot()
    t += dt
    sim.run_until(t)
