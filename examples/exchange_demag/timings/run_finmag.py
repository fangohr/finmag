import commands
import os, time
import numpy as np
import dolfin as df
from finmag.sim.llg import LLG
from finmag.sim.integrator import LLGIntegrator
from finmag.util.timings import timings
from finmag.util.convert_mesh import convert_mesh

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create meshes
h5mesh = '- netgen -geofile=bar.geo -meshfiletype="Neutral Format" -meshfile=bar.neutral -batchmode'
xmlmesh = '- netgen -geofile=bar.geo -meshfiletype="DIFFPACK Format" -meshfile=bar.grid -batchmode'
nmeshimport = 'nmeshimport --netgen bar.neutral bar.nmesh.h5'
dolfinconvert = 'dolfin-convert bar.grid bar.xml && gzip -f bar.xml'
if not os.path.isfile(MODULE_DIR + "/bar.nmesh.h5"):
    commands.getstatusoutput(h5mesh)
    commands.getstatusoutput(nmeshimport)
if not os.path.isfile(MODULE_DIR + "/bar.xml.gz"):
    commands.getstatusoutput(xmlmesh)
    commands.getstatusoutput(dolfinconvert)

# Run nmag
commands.getstatusoutput("nsim run_nmag.py --clean")

# Run finmag
setupstart = time.time()

# Set up LLG
mesh = df.Mesh(MODULE_DIR + "/bar.xml.gz")
llg = LLG(mesh, mesh_units=1e-9)
llg.Ms = 0.86e6
llg.A = 13.0e-12
llg.alpha = 0.5
llg.set_m((1,0,1))
llg.setup(use_exchange=True, use_dmi=False, use_demag=True, demag_method="FK")

# Set up time integrator
integrator = LLGIntegrator(llg, llg.m)
times = np.linspace(0, 3.0e-10, 61)

dynamicsstart = time.time()
for t in times:
    # Integrate
    integrator.run_until(t)
endtime = time.time()

output = open(MODULE_DIR + "/results.rst", "a")
output.write("\nFinmag results:\n")
output.write("---------------\n")
output.write("Setup: %.3f sec.\n" % (dynamicsstart-setupstart))
output.write("Dynamics: %.3f sec.\n" % (endtime-dynamicsstart))
output.write("\nFinmag details:\n")
output.write(str(timings))
output.close()

# Cleanup
files = ["bar_bi.xml", "bar.grid", "bar_mat.xml", "bar.neutral", "bar.xml.bak", "run_nmag_log.log", "bar_dat.ndt"]
for file in files:
    if os.path.isfile(MODULE_DIR + '/' + file):
        os.remove(MODULE_DIR + '/' + file)
