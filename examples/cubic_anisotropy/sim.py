import math
import numpy as np
from finmag import Simulation
from finmag.energies import CubicAnisotropy, Demag, Exchange, Zeeman
from finmag.util.consts import flux_density_to_field_strength
from finmag.util.meshes import cylinder

ps = 1e-12

# Mesh

mesh = cylinder(r=10, h=2.5, maxh=3.0, filename='disk')
unit_length = 1e-9

# Material Definition

Ms = 9.0e5  # saturation magnetisation in A/m
A = 2.0e-11  # exchange constant in J/m
alpha = 0.01  # damping constant no unit
gamma = 2.3245e5  # m / (As)
u1 = (1, 0, 0)  # cubic anisotropy axes
u2 = (0, 1, 0)
K1 = -1e4  # anisotropy constant in J/m^3

# External Field

# the field will be zero, is this intended?
H_app_dir = np.array((0, 0, 0))
# converts Tesla to A/m (divides by mu0)
H_app_strength = flux_density_to_field_strength(1e-3)

# Spin-Polarised Current

current_density = 100e10  # in A/m^2
polarisation = 0.76
thickness = 2.5e-9  # in m

theta = math.pi
phi = math.pi / 2
direction = (math.sin(theta) * math.cos(phi),
             math.sin(theta) * math.sin(phi),
             math.cos(theta))

# Create Simulation

sim = Simulation(mesh, Ms, unit_length, name='disksim')
sim.alpha = alpha
sim.gamma = gamma
sim.set_m((0.01, 0.01, 1.0))
sim.set_stt(current_density, polarisation, thickness, direction)
sim.add(Demag())
sim.add(Zeeman(H_app_strength * H_app_dir))
sim.add(Exchange(A))
sim.add(CubicAnisotropy(u1, u2, K1))
sim.set_tol(reltol=1e-8, abstol=1e-8)  # timestepper tolerances

sim.schedule('save_m', every=10*ps)
sim.schedule('save_averages', every=100*ps)
sim.run_until(2000 * ps)
