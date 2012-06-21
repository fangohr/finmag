import os
import dolfin as df
import numpy as np
from finmag.util.convert_mesh import convert_mesh
from finmag.sim.helpers import quiver

MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"
mesh_file = MODULE_DIR + "bar.geo"
initial_m_file = MODULE_DIR + "m_init.txt"
zero_crossing_m_file = MODULE_DIR + "m_zero.txt"

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
