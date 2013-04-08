import math
from nmag.common import SI, degrees_per_ns, Tesla, mu0, at, every
from nmag.nmag5 import Simulation, MagMaterial

# Applied field
Happ_dir = [0, 0, 10] # in mT

# Material definition
mat = MagMaterial("Py", Ms=SI(860e3, "A/m"),
  exchange_coupling=SI(13e-12, "J/m"),
  llg_gamma_G=SI(221017, "m/s A"),
  llg_damping=SI(0.014))

sim = Simulation("nmag_no_stt", do_demag=False)
nm = SI(1e-9, "m")
sim.load_mesh("mesh.nmesh", [("cube", mat)], unit_length=nm)
sim.set_m([1, 0.01, 0.01])
sim.set_H_ext(Happ_dir, 0.001*Tesla/mu0)

# Define the tolerances for the simulation
ns = SI(1e-9, "s")
sim.set_params(stopping_dm_dt=0*degrees_per_ns,
               ts_rel_tol=1e-8, ts_abs_tol=1e-8,
               ts_pc_rel_tol=1e-3, ts_pc_abs_tol=1e-8,
               demag_dbc_rel_tol=1e-6, demag_dbc_abs_tol=1e-6)
sim.relax(save=[("averages", every("time", 0.01*ns))],
          do=[("exit", at("time", 10*ns))])
