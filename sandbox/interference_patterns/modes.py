import numpy as np
import dolfin as df
import matplotlib.pyplot as plt
from math import pi
from finmag import Simulation
from finmag.energies import Zeeman, UniaxialAnisotropy, ThinFilmDemag

# todo: show trajectories in 3D space
# todo: what is the fundamental frequency?

mesh = df.Interval(1, 0, 1e-9)
# not possible to have 0D mesh. Tried PolygonalMeshGenerator and MeshEditor.

Ms_Oersted = 17.8e3; Ms = Ms_Oersted / (pi * 4e-3);
H_anis_Oersted = 986; H_anis = H_anis_Oersted / (pi * 4e-3);
for H_ext_Oersted in [1e3, 1.5e3, 2e3]:
    J_rel = []; inv_t0_over_J_rel = [];
    for J_mult in np.linspace(1.0, 1.5, 20):
        H_ext = H_ext_Oersted / (pi * 4e-3)

        sim = Simulation(mesh, Ms)
        sim.set_m((0.1, 0.1, 1.0)) # "nearly parallel to fixed layer magnetisation"
        sim.alpha = 0.003
        sim.add(Zeeman((0, 0, - H_ext)))
        sim.add(UniaxialAnisotropy(K1=H_anis, axis=(0, 0, 1)))
        sim.add(ThinFilmDemag("x", 0.65))

        J0 = -1e12 # A/m^2 or 10^8 A/cm^2
        sim.set_stt(df.Constant(J_mult*J0), polarisation=1.0, thickness=4e-9, direction=(0, 0, 1))

        t = 0.0; dt = 1e-12; t_max = 1e-9;
        while t <= t_max:
            mz = sim.m.reshape((3, -1)).mean(1)[2]
            if mz <= 0:
                J_rel.append(J_mult - 1)
                inv_t0_over_J_rel.append(1/t)
                break
            t += dt
            sim.run_until(t)
        plt.plot(J_rel, inv_t0_over_J_rel) 
plt.show()

