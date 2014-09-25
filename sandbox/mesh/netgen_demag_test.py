import dolfin as df
import finmag
import numpy as np
from finmag.energies import Demag
from finmag import Simulation as Sim
from finmag.util.mesh_templates import Nanodisk

# #############################################
# ###### MESH GENERATOR

# LENGTHS
lex = 5.76  # nm (Exchange length)

# The length of the nanotube. when using 1100 is alright, but
# starting from ~ 1150, the memory blows up
L = 2000.  # nm

r_ext = 3 * lex  # External diameter in nm
r_int = 0.8 * r_ext

# Define the geometric structures to be processed
ntube_e = Nanodisk(d=2 * r_ext, h=L, center=(0, 0, 0),
                   valign='bottom', name='ntube_e')
# Internal cylinder to be substracted:
ntube_i = Nanodisk(d=2 * r_int, h=L, center=(0, 0, 0),
                   valign='bottom', name='ntube_i')
# Substract the small cylinder
ntube = ntube_e - ntube_i

# Create the mesh
mesh = ntube.create_mesh(maxh=4.0, save_result=False)


# #############################################
# ###### SIMULATION

# MATERIAL PARAMETERS
Ms = 7.96e5  # A m**-1 saturation magnetisation

# Inititate the simulation object with the lengths in nm
sim = Sim(mesh, Ms, unit_length=1e-9)

sim.add(Demag())
