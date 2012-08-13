import os, time
import nmag
from nmag import SI

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
setupstart=time.time()

mat_Py = nmag.MagMaterial(name="Py",
                          Ms=SI(0.86e6,"A/m"),
                          exchange_coupling=SI(13.0e-12, "J/m"),
                          llg_damping=0.5)

sim = nmag.Simulation("bar")

sim.load_mesh(os.path.join(MODULE_DIR, "bar.nmesh.h5"),
              [("Py", mat_Py)],
              unit_length=SI(1e-9,"m"))

sim.set_m([1,0,1])

dt = SI(5e-12, "s")

sim.save_data()

dynamicsstart=time.time()
for i in range(0, 61):
    sim.advance_time(dt*i)

endtime = time.time()

output = open(os.path.join(MODULE_DIR, "results.rst"), "w")
output.write("Nmag results:\n")
output.write("-------------\n")
output.write("Setup: %.3f sec.\n" % (dynamicsstart-setupstart))
output.write("Dynamics: %.3f sec.\n" % (endtime-dynamicsstart))
output.close()
