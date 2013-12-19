import os, sys, math
from nsim.netgen import netgen_mesh_from_string
from nmag.common import SI, degrees_per_ns, Tesla, mu0, at, every
from nmag.nmag5 import Simulation, MagMaterial, cubic_anisotropy
from nsim.model import Value

ps = SI(1e-12, "s"); nm = SI(1e-9, "m")               # Useful definitions
theta_rad = 3.141592654
phi_rad = 1.570796327                             
#length, width, thick = (2*nm, 16*nm, 64*nm)       # System geometry
current_density = SI( 100e10, "A/m^2") # Applied current
Happ_dir = [0, 0, 0]                           # Applied field (mT)- 

# Material definition
anis = cubic_anisotropy(axis1=[1, 0, 0], axis2=[0,1,0], K1=SI(-1e4, "J/m^3"))
mat = MagMaterial("Co",
                  Ms=SI(9.0e5, "A/m"),
                  exchange_coupling=SI(2.0e-11, "J/m"),
                  llg_gamma_G=SI(2.3245e5, "m/s A"),
                  llg_damping=SI(0.01),
                  anisotropy = anis)

mat.sl_P = 0.76             # Polarisation
mat.sl_lambda = 2.0        # lambda parameter
mat.sl_d = SI(2.5e-9, "m") # Free layer thickness

sim = Simulation(do_sl_stt=True, do_demag=False)
sim.load_mesh("disc.nmesh.h5", [("region1", mat)], unit_length=nm)
sim.set_m([0.01, 0.01, 1 ])
sim.set_H_ext(Happ_dir, 0.001*Tesla/mu0)

# Direction of the polarization

P_direction = [math.sin(theta_rad)*math.cos(phi_rad),
               math.sin(theta_rad)*math.sin(phi_rad),
               math.cos(theta_rad)]

# Set the polarization direction and current density
sim.model.quantities["sl_fix"].set_value(Value(P_direction))
sim.model.quantities["sl_current_density"].set_value(Value(current_density))

# Define the tolerances for the simulation
sim.set_params(stopping_dm_dt=0*degrees_per_ns,
               ts_rel_tol=1e-8, ts_abs_tol=1e-8,
               ts_pc_rel_tol=1e-3, ts_pc_abs_tol=1e-8,
               demag_dbc_rel_tol=1e-6, demag_dbc_abs_tol=1e-6)
sim.relax(save=[('fields', at('convergence') | every("time", 10*ps)),('averages', every('time', SI(0.1e-9, "s")) | at('stage_end'))],
          do=[("exit", at("time", 2000*ps))])
