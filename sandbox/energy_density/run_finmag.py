import numpy as np
import pylab as p
from dolfin import Mesh
from finmag.sim.llg import LLG
from finmag.drivers.llg_integrator import llg_integrator

# Create mesh
mu = 1e-9
mesh = Mesh("coarse_bar.xml.gz")
#mesh = Mesh("bar.xml.gz")

# Setup LLG
llg = LLG(mesh, unit_length=mu)
llg.Ms = 0.86e6
llg.A = 13.0e-12
llg.alpha = 0.5
llg.set_m((1,0,1))
llg.setup(use_exchange=True, use_dmi=False, use_demag=True, demag_method="FK")

# Set up time integrator
integrator = llg_integrator(llg, llg.m)
dt = 5e-12

######
# After ten time steps, plot the energy density
# from z=0 to z=100 through the center of the body.
######

# Integrate
integrator.run_until(dt*10)
exch = llg.exchange.energy_density_function()
demag = llg.demag.energy_density_function()
finmag_exch, finmag_demag = [], []
R = range(100)
for i in R:
    finmag_exch.append(exch([15, 15, i]))
    finmag_demag.append(demag([15, 15, i]))

# Read nmag data
nmag_exch = [float(i) for i in open("nmag_exch_Edensity.txt", "r").read().split()]
nmag_demag = [float(i) for i in open("nmag_demag_Edensity.txt", "r").read().split()]

# Read oommf data
oommf_exch = np.genfromtxt("oommf_exch_Edensity.txt")
oommf_demag = np.genfromtxt("oommf_demag_Edensity.txt")
oommf_coords = np.genfromtxt("oommf_coords_z_axis.txt") * 1e9

# Plot exchange energy density
p.plot(R, finmag_exch, 'o-', R, nmag_exch, 'x-', oommf_coords, oommf_exch, "+-")
p.xlabel("nm")
p.title("Exchange energy density")
p.legend(["Finmag", "Nmag", "oommf"])
p.savefig("exch.png")

# Plot demag energy density
p.figure()
p.plot(R, finmag_demag, 'o-', R, nmag_demag, 'x-', oommf_coords, oommf_demag, "+-")
p.xlabel("nm")
p.title("Demag energy density")
p.legend(["Finmag", "Nmag", "oommf"])
p.savefig("demag.png")

#p.show()
print "Plots written to exch.png and demag.png"
