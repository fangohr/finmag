from mshr import Cylinder
import dolfin as df
import mshr
import finmag
import numpy as np
from finmag.energies import Demag
from finmag import Simulation as Sim


# #############################################
# ###### MESH GENERATOR

# LENGTHS
lex = 5.76  # nm (Exchange length)
L = 2000.  # nm
r_ext = 3 * lex  # External diameter in nm
r_int = 0.8 * r_ext

# Define the geometric structures to be processed
ntube = (Cylinder(df.Point(0., 0., 0.),
                  df.Point(0., 0., L),
                  r_ext,
                  r_ext,
                  )
         -
         Cylinder(df.Point(0., 0., 0.),
                  df.Point(0., 0., L),
                  r_int,
                  r_int,
                  )
         )

# Create the mesh
diag = np.sqrt(L ** 2 + (3 * lex) ** 2)
resolut = diag / 3.
mesh = mshr.generate_mesh(ntube, resolut)

# Output Mesh
outp = file('mesh_info.dat', 'w')
outp.write(finmag.util.meshes.mesh_info(mesh))

# Save mesh to file
file = df.File("nanotube.xml")
file << mesh

# #############################################
# ###### SIMULATION

# Import mesh from the 'xml' file produced by the mesh generator script
mesh = df.Mesh('nanotube.xml')

# MATERIAL PARAMETERS
Ms = 7.96e5  # A m**-1 saturation magnetisation

# Inititate the simulation object with the lengths in nm
sim = Sim(mesh, Ms, unit_length=1e-9)

sim.add(Demag())
