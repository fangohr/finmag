import os, time
import dolfin as df
from finmag.util.timings import default_timer
from finmag import Simulation
from finmag.energies import Exchange, Demag

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# Run the nmag2 example on PETSc and PETScCUSP and compare timings

def run_finmag():
	# Setup
	setupstart = time.time()
        mesh = df.Box(0,0,0,30,30,100,15,15,50)
	sim = Simulation(mesh, Ms=0.86e6, unit_length=1e-9)
	sim.set_m((1, 0, 1))
	demag = Demag("FK")
        demag.parameters["poisson_solver"]["method"] = "cg"
        demag.parameters["poisson_solver"]["preconditioner"] = "jacobi"
	demag.parameters["laplace_solver"]["method"] = "cg"
	demag.parameters["laplace_solver"]["preconditioner"] = "bjacobi"
	sim.add(demag)
	sim.add(Exchange(13.0e-12))

	# Dynamics
	dynamicsstart = time.time()
	sim.run_until(3.0e-10)
	endtime = time.time()

	# Write output to results.txt
	output = open(os.path.join(MODULE_DIR, "results.txt"), "a")
        output.write("\nBackend %s:\n" % df.parameters["linear_algebra_backend"])
	output.write("\nSetup: %.3f sec.\n" % (dynamicsstart-setupstart))
	output.write("Dynamics: %.3f sec.\n\n" % (endtime-dynamicsstart))
	output.write(str(default_timer))
	output.close()

# Need a clean file
if os.path.isfile(os.path.join(MODULE_DIR, "results.txt")):
    os.remove(os.path.join(MODULE_DIR, "results.txt"))

df.parameters["linear_algebra_backend"] = "PETSc"
run_finmag()

df.parameters["linear_algebra_backend"] = "PETScCusp"
run_finmag()
