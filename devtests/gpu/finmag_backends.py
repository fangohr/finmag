import commands
import os, time
import numpy as np
import dolfin as df
from finmag.util.timings import timings
from finmag.util.convert_mesh import convert_mesh

from finmag import Simulation
from finmag.energies import Exchange, Demag


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

####
# Run the nmag2 example on PETSc and PETScCUSP and compare timings
####

def run_finmag():
	# Setup
	setupstart = time.time()
        #mesh = df.Mesh(convert_mesh(MODULE_DIR + "/bar.geo"))
        mesh = df.Box(0,0,0,30,30,100,18,18,60)
	sim = Simulation(mesh, Ms=0.86e6, unit_length=1e-9)
	sim.set_m((1, 0, 1))
	demag = Demag("FK_magpar")
        #demag.parameters["poisson_solver"]["method"] = "gmres"
        #demag.parameters["poisson_solver"]["preconditioner"] = "bjacobi"
	#demag.parameters["laplace_solver"]["method"] = "gmres"
	#demag.parameters["laplace_solver"]["preconditioner"] = "sor"
	sim.add(demag)
	sim.add(Exchange(13.0e-12))

	# Dynamics
	dynamicsstart = time.time()
	sim.run_until(3.0e-10)
	endtime = time.time()

	# Write output to results.rst
	output = open(MODULE_DIR + "/results.txt", "a")
        output.write("\nBackend: %s:\n" % df.parameters["linear_algebra_backend"])
	output.write("---------------\n")
	output.write("Setup: %.3f sec.\n" % (dynamicsstart-setupstart))
	output.write("Dynamics: %.3f sec.\n" % (endtime-dynamicsstart))
	output.write("\nFinmag details:\n")
	output.write(str(timings))
	output.close()

# Need a clean file
if os.path.isfile(MODULE_DIR + "/results.txt"):
    os.remove(MODULE_DIR + "/results.txt")

df.parameters["linear_algebra_backend"] = "PETSc"
run_finmag()

df.parameters["linear_algebra_backend"] = "PETScCusp"
run_finmag()
