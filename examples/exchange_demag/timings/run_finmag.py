import commands
import os, time
import dolfin as df
from finmag.util.timings import default_timer
from finmag.util.meshes import from_geofile

from finmag import Simulation
from finmag.energies import Exchange, Demag

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

####
# Run the nmag2 example on nmag and finmag and compare timings
####

# Create meshes
neutralmesh = 'netgen -geofile=bar.geo -meshfiletype="Neutral Format" -meshfile=bar.neutral -batchmode'
nmeshimport = 'nmeshimport --netgen bar.neutral bar.nmesh.h5'
if not os.path.isfile(os.path.join(MODULE_DIR, "bar.nmesh.h5")):
    commands.getstatusoutput(neutralmesh)
    commands.getstatusoutput(nmeshimport)
if not os.path.isfile(os.path.join(MODULE_DIR, "bar.xml.gz")):
    from_geofile(os.path.join(MODULE_DIR, "bar.geo"))

# Run nmag
commands.getstatusoutput("nsim run_nmag.py --clean")

# Run finmag
# setupstart = time.time()
# Set up LLG
#mesh = df.Mesh(os.path.join(MODULE_DIR, "bar.xml.gz"))
#llg = LLG(mesh, unit_length=1e-9)
#llg.Ms = 0.86e6
#llg.A = 13.0e-12
#llg.alpha = 0.5
#llg.set_m((1,0,1))
#llg.setup(use_exchange=True, use_dmi=False, use_demag=True, demag_method="FK")

# Setup
setupstart = time.time()
mesh = df.Mesh(os.path.join(MODULE_DIR, "bar.xml.gz"))
sim = Simulation(mesh, Ms=0.86e6, unit_length=1e-9)
sim.set_m((1, 0, 1))
demag = Demag()
demag.parameters["poisson_solver"]["method"] = "minres"
demag.parameters["laplace_solver"]["method"] = "gmres"
demag.parameters["laplace_solver"]["preconditioner"] = "sor"
sim.add(demag)
sim.add(Exchange(13.0e-12))

# Dynamics
dynamicsstart = time.time()
sim.run_until(3.0e-10)
endtime = time.time()

# Write output to results.rst
output = open(os.path.join(MODULE_DIR, "results.rst"), "a")
output.write("\nFinmag results:\n")
output.write("---------------\n")
output.write("Setup: %.3f sec.\n" % (dynamicsstart-setupstart))
output.write("Dynamics: %.3f sec.\n" % (endtime-dynamicsstart))
output.write("\nFinmag details:\n")
output.write("---------------\n")
output.write("\nFinmag solver parameters:\n")
output.write("%s \n"%repr(demag.parameters.to_dict()))
output.write("\nFinmag solver tolerances:")
output.write("\nFirst linear solve :%s" %(demag.demag._poisson_solver.parameters.to_dict()["relative_tolerance"]))
output.write("\nSecond linear solve: %s \n \n"% (demag.demag._laplace_solver.parameters.to_dict()["relative_tolerance"]))
output.write(str(default_timer))
output.close()

# Cleanup
files = ["bar_bi.xml", "bar.grid", "bar_mat.xml", "bar.neutral", "bar.xml.bak", "run_nmag_log.log", "bar_dat.ndt"]
for file in files:
    fname = os.path.join(MODULE_DIR, file)
    if os.path.isfile(fname):
        os.remove(fname)
