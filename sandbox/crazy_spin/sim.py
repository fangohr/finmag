import dolfin as df
import numpy as np
import pylab as plt
from finmag import Simulation as Sim
from finmag.energies import Exchange, DMI, Zeeman
from finmag.util.meshes import nanodisk
from finmag.util.consts import mu0

def skyrm_init(x):
    r = (x[0]**2 + x[1]**2)**0.5
    if r < 50e-9:
        return (0, 0, -1)
    else:
        return (0, 0, 1)

d = 30  # nm
thickness = 5  # nm
hmax = 3  # nm
Ms = 3.84e5  # A/m
A = 8.78e-12  # J/m 
D = 1.58e-3  # J/m**2
H = (0, 0, 0)
alpha = 1e-4

t_exc = 0.5e-9
n_exc = 201
H_exc_max = 5e-3 / mu0
fc = 20e9

t_end = 5e-9
delta_t = 5e-12
t_array = np.arange(0, t_end, delta_t)

filename = 'test.npz'
mesh_filename = 'mesh_test.xml'

mesh = nanodisk(d, thickness, hmax, save_result=False)

sim = Sim(mesh, Ms, unit_length=1e-9)
sim.set_m((0, 0, 1))

sim.add(Exchange(A))
sim.add(DMI(D))
sim.add(Zeeman(H))

# Ground state simulation
sim.relax(stopping_dmdt=1e-5)
sim.reset_time(0)

n_nodes = mesh.num_vertices()
m_gnd = np.zeros(3*n_nodes)

m = np.zeros((len(t_array), 3*n_nodes))
m_gnd = sim.llg._m.vector().array()

# Excitation
t_excitation = np.linspace(0, t_exc, n_exc)
H_exc = H_exc_max * np.sinc(2*np.pi*fc*(t_excitation-t_exc/2))

#plt.plot(t_excitation, H_exc)
#plt.show()

sim.alpha = alpha

for i in xrange(len(t_excitation)):
    sim.set_H_ext((0, 0, H_exc[i]))
    sim.run_until(t_excitation[i])
sim.reset_time(0)

sim.set_H_ext(H)
# Dynamics
for i in xrange(len(t_array)):
    sim.run_until(t_array[i])
    df.plot(sim.llg._m)
    m[i, :] = sim.llg._m.vector().array()
    sim.save_vtk('vtk_file_' + str(i) + '.pvd', overwrite=True)

mesh_file = df.File(mesh_filename)
mesh_file << mesh

np.savez('test.npz', m_gnd=m_gnd, m=m, delta_t=np.array([delta_t]))
