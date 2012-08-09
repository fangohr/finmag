import os
import dolfin as df
import numpy as np
from finmag.util.convert_mesh import convert_mesh
from finmag.util.helpers import quiver, read_float_data

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
mesh_file = os.path.join(MODULE_DIR, "bar.geo")
initial_m_file = os.path.join(MODULE_DIR, "m_init.txt")
zero_crossing_m_file = os.path.join(MODULE_DIR, "m_zero.txt")
averages_m_file = os.path.join(MODULE_DIR, "m_averages.txt")
averages_m_file_martinez = os.path.join(MODULE_DIR, "m_averages_martinez.txt")

# plots of average magnetisation components

averages = np.array(read_float_data(averages_m_file))
averages_martinez = np.array(read_float_data(averages_m_file_martinez))

print averages_martinez.shape

import matplotlib.pyplot as plt

times_martinez = averages_martinez[::5,0]
mx_martinez = averages_martinez[::5,1]
my_martinez = averages_martinez[::5,2]
mz_martinez = averages_martinez[::5,3]
plt.plot(times_martinez, mx_martinez, "-", color="0.6", label="$m_x\,\mathrm{Martinez\, et\, al.}$")
plt.plot(times_martinez, my_martinez, "--", color="0.6", label="")
plt.plot(times_martinez, mz_martinez, ":", color="0.6", label="")

times = averages[:,0] * 1e9
mx = averages[:,1]
my = averages[:,2]
mz = averages[:,3]
plt.plot(times, mx, "b-", label="$m_x\,\mathrm{FinMag}$")
plt.plot(times, my, "r--", label="$m_y$")
plt.plot(times, mz, ":", color="0.1", label="$m_z$")

plt.xlabel("$\mathrm{time}\, (\mathrm{ns})$")
plt.ylabel("$<m_i> = <M_i>/M_\mathrm{S}$")
plt.legend()
plt.xlim([0,2])
plt.savefig(os.path.join(MODULE_DIR, "m_averages.pdf"))

import sys; sys.exit()
# 3D plots for magnetisation at t0 and at zero crossing

mesh = df.Mesh(convert_mesh(mesh_file))
m0_np = np.loadtxt(initial_m_file)
m1_np = np.loadtxt(zero_crossing_m_file)

S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
m0 = df.Function(S3)
m0.vector()[:] = m0_np
m1 = df.Function(S3)
m1.vector()[:] = m1_np

xs = np.linspace(0, 500, 30)
ys = np.linspace(0, 125, 10)
z = 1.5
nb_nodes = len(xs) * len(ys)

reduced_mesh_coordinates = np.zeros((nb_nodes, 3))
reduced_m0 = np.zeros((3, nb_nodes))
reduced_m1 = np.zeros((3, nb_nodes))

i = 0
for x in xs:
    for y in ys:
        reduced_mesh_coordinates[i] = [x, y, z]

        m0_x, m0_y, m0_z = m0(x, y, z)
        m1_x, m1_y, m1_z = m1(x, y, z)

        reduced_m0[0][i] = m0_x
        reduced_m0[1][i] = m0_y
        reduced_m0[2][i] = m0_z

        reduced_m1[0][i] = m1_x
        reduced_m1[1][i] = m1_y
        reduced_m1[2][i] = m1_z

        i += 1

quiver(reduced_m0.flatten(), reduced_mesh_coordinates)
quiver(reduced_m1.flatten(), reduced_mesh_coordinates)
