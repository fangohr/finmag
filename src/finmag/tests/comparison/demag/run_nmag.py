import os
import numpy as np
import nmag
from nmag import SI

MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

#create simulation object
sim = nmag.Simulation()

# define magnetic material
Py = nmag.MagMaterial(name = 'Py', Ms = SI(1.0, 'A/m'))

# load mesh
sim.load_mesh("sphere.nmesh.h5", [('main', Py)], unit_length = SI(1e-9, 'm'))

# set initial magnetisation
sim.set_m([1,0,0])

# set external field
sim.set_H_ext([0,0,0], SI('A/m'))

H = sim.get_subfield('H_demag')
np.savetxt("H_demag_nmag.txt", H)
