import math
from nmag.common import SI, degrees_per_ns, Tesla, mu0, at, every
from nmag.nmag5 import Simulation, MagMaterial
from nsim.model import Value

# Geometry
nm = SI(1e-9, "m")
length = width = 12.5*nm; height = 2.5*nm;

# Applied field
Happ_dir = [0, 0, 10] # in mT

# Material definition
mat = MagMaterial("Py", Ms=SI(860e3, "A/m"),
  exchange_coupling=SI(13e-12, "J/m"),
  llg_gamma_G=SI(221017, "m/s A"),
  llg_damping=SI(0.014))

# Parameters relevant to spin-torque transfer
# Current and Current Density
I = SI(5e-5, "A")
J = I/(length * width)

mat.sl_P = 0.4             # Polarisation
mat.sl_lambda = 2.0        # lambda parameter
mat.sl_d = height          # Free layer thickness

# Direction of the polarisation
theta = 40.0
phi = 90.0
theta_rad = math.pi*theta/180.0
phi_rad = math.pi*phi/180.0
P_direction = [math.sin(theta_rad)*math.cos(phi_rad),
               math.sin(theta_rad)*math.sin(phi_rad),
               math.cos(theta_rad)]

sim = Simulation(do_sl_stt=True, do_demag=False)
sim.load_mesh("mesh.nmesh", [("cube", mat)], unit_length=nm)
sim.set_m([1, 0.01, 0.01])
sim.set_H_ext(Happ_dir, 0.001*Tesla/mu0)

# Set the polarization direction and current density
sim.model.quantities["sl_fix"].set_value(Value(P_direction))
sim.model.quantities["sl_current_density"].set_value(Value(J))

# Define the tolerances for the simulation
ps = SI(1e-12, "s")
sim.set_params(stopping_dm_dt=0*degrees_per_ns,
               ts_rel_tol=1e-8, ts_abs_tol=1e-8,
               ts_pc_rel_tol=1e-3, ts_pc_abs_tol=1e-8,
               demag_dbc_rel_tol=1e-6, demag_dbc_abs_tol=1e-6)
sim.relax(save=[("averages", every("time", 10*ps))],
          do=[("exit", at("time", 10000*ps))])
