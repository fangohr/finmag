import time
import numpy as np
import dolfin as df
from finmag.sim.llg import LLG
from finmag.sim.integrator import LLGIntegrator
from finmag.util.timings import timings
from finmag.util.convert_mesh import convert_mesh

setupstart = time.time()

# Set up LLG
mesh = df.Mesh("bar.xml.gz")
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

output = open("results.txt", "a")
output.write("\nFinmag results:\n")
output.write("Setup: %.3f sec.\n" % (dynamicsstart-setupstart))
output.write("Dynamics: %.3f sec.\n" % (endtime-dynamicsstart))
output.write("\nFinmag details:\n")
output.write(str(timings))
output.close()
