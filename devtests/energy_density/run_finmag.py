from dolfin import Mesh
from finmag.sim.llg import LLG
from finmag.sim.integrator import LLGIntegrator

# Create mesh
mesh_units = 1e-9
mesh = Mesh("coarse_bar.xml.gz")

# Setup LLG
llg = LLG(mesh, mesh_units=mesh_units)
llg.Ms = 0.86e6
llg.A = 13.0e-12
llg.alpha = 0.5
llg.set_m((1,0,1))
llg.setup(use_exchange=True, use_dmi=False, use_demag=True, demag_method="FK")

# Set up time integrator
integrator = LLGIntegrator(llg, llg.m)

dt = 5e-12

# Integrate
integrator.run_until(dt*10)
E1 = llg.exchange.energy_density()
E2 = llg.demag.energy_density()
print E2

