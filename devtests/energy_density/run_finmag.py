import pylab as p
from dolfin import Mesh
from finmag.sim.llg import LLG
from finmag.sim.integrator import LLGIntegrator

# Create mesh
mu = 1e-9
mesh = Mesh("coarse_bar.xml.gz")

# Setup LLG
llg = LLG(mesh, mesh_units=mu)
llg.Ms = 0.86e6
llg.A = 13.0e-12
llg.alpha = 0.5
llg.set_m((1,0,1))
llg.setup(use_exchange=True, use_dmi=False, use_demag=True, demag_method="FK")

# Set up time integrator
integrator = LLGIntegrator(llg, llg.m)

dt = 5e-12

######
# After ten time steps, plot the energy density
# from z=0 to z=100 through the center of the body.
######


# Integrate
integrator.run_until(dt*10)
#E1 = llg.exchange.energy_density()
#E2 = llg.demag.energy_density()
exch = llg.exchange.density_function()
demag = llg.demag.density_function()
finmag_exch, finmag_demag = [], []
R = range(0, 110, 10)
for i in R:
    finmag_exch.append(exch([15, 15, i]))
    finmag_demag.append(demag([15, 15, i]))

# Read nmag data
nmag_exch = [float(i) for i in open("nmag_exch_Edensity.txt", "r").read().split()]
nmag_demag = [float(i) for i in open("nmag_demag_Edensity.txt", "r").read().split()]

# Plot
p.plot(R, finmag_exch)#, R, nmag_exch)
p.xlabel("nm")
p.title("Exchange energy density")
p.legend(["Finmag"])#, "Nmag"]))
p.savefig("finmag_exch.png")

p.figure()
p.plot(R, nmag_exch)
p.xlabel("nm")
p.title("Exchange energy density")
p.legend(["Nmag"])
p.savefig("nmag_exch.png")

p.figure()
p.plot(R, finmag_demag)#, R, nmag_demag)
p.xlabel("nm")
p.title("Demag energy density")
p.legend(["Finmag"])#, "Nmag"]))
p.savefig("finmag_demag.png")

p.figure()
p.plot(R, nmag_demag)
p.xlabel("nm")
p.title("Demag energy density")
p.legend(["Nmag"])
p.savefig("nmag_demag.png")

print "Plots written to finmag_exch.png, nmag_exch.png, finmag_demag.png and nmag_demag.png"
#p.show()
