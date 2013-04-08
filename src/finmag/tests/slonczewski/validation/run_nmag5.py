import os, math
import numpy as np
from nsim.netgen import netgen_mesh_from_string
from nmag.common import SI, degrees_per_ns, Tesla, mu0, at, every
from nmag.nmag5 import Simulation, MagMaterial, uniaxial_anisotropy
from nsim.model import Value

ps = SI(1e-12, "s"); nm = SI(1e-9, "m")               # Useful definitions
theta = 40.0; phi = 90.0                              # Polarization direction
length, width, thick = (12.5*nm, 12.5*nm, 5*nm)       # System geometry
I = SI(5e-5, "A"); current_density = I/(length*width) # Applied current
Happ_dir = [0.2, 0.2, 10.0]                           # Applied field (mT)

# Create the mesh, if it does not exist
mesh_filename = "film.nmesh.h5"
if not os.path.exists(mesh_filename):
  mesh_geo = \
    ("algebraic3d\n"
     "solid cube = orthobrick (0, 0, 0; %s, %s, %s) -maxh = 2.5;\n"
     "tlo cube;\n" % tuple(map(lambda x: float(x/nm), (length, width, thick))))
  netgen_mesh_from_string(mesh_geo, mesh_filename, keep_geo=True)

# Material definition
anis = uniaxial_anisotropy(axis=[0, 0, 1], K1=-SI(0.1e6, "J/m^3"))
mat = MagMaterial("Py",
                  Ms=SI(860e3, "A/m"),
                  exchange_coupling=SI(13e-12, "J/m"),
                  llg_gamma_G=SI(221017, "m/s A"),
                  llg_damping=SI(0.014),
                  anisotropy=anis)

mat.sl_P = 0.4             # Polarisation
mat.sl_lambda = 2.0        # lambda parameter
mat.sl_d = SI(5.0e-9, "m") # Free layer thickness

sim = Simulation(do_sl_stt=True, do_demag=False)
sim.load_mesh(mesh_filename, [("region1", mat)], unit_length=nm)
sim.set_m([1, 0.01, 0.01])
sim.set_H_ext(Happ_dir, 0.001*Tesla/mu0)

# Direction of the polarization. We normalize this by hand because
# nmag doesn't seem to do it automatically.
theta_rad = math.pi*theta/180.0
phi_rad = math.pi*phi/180.0
P_direction = np.array([math.sin(theta_rad)*math.cos(phi_rad),
                        math.sin(theta_rad)*math.sin(phi_rad),
                        math.cos(theta_rad)])
P_direction = list(P_direction / np.linalg.norm(P_direction))

# Set the polarization direction and current density
sim.model.quantities["sl_fix"].set_value(Value(P_direction))
sim.model.quantities["sl_current_density"].set_value(Value(current_density))

# Define the tolerances for the simulation
sim.set_params(stopping_dm_dt=0*degrees_per_ns,
               ts_rel_tol=1e-8, ts_abs_tol=1e-8,
               ts_pc_rel_tol=1e-3, ts_pc_abs_tol=1e-8,
               demag_dbc_rel_tol=1e-6, demag_dbc_abs_tol=1e-6)
sim.relax(save=[("averages", every("time", 5*ps))],
          do=[("exit", at("time", 10000*ps))])
