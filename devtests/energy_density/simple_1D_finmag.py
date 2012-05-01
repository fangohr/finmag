from dolfin import *
from finmag.sim.llg import LLG
import numpy as np
import pylab as p
import commands

commands.getstatusoutput("nsim simple_1D_nmag.py --clean")

mesh = Interval(100, 0, 10e-9)
llg = LLG(mesh)
llg.Ms = 1
llg.set_m(("cos(x[0]*pi/10e-9)", "sin(x[0]*pi/10e-9)", "0"))
llg.setup(use_exchange=True, use_dmi=False, use_demag=False)

finmag_data = llg.exchange.energy_density()
nmag_data = np.load("nmag_hansconf.npy")


print "Finmag:"
print finmag_data
print "Nmag:"
print nmag_data

print "Finmag max-min:", max(finmag_data) - min(finmag_data)
print "Nmag max-min:", max(nmag_data) - min(nmag_data)

x = np.linspace(0, 10e-9, 101)
p.plot(x, finmag_data, x, nmag_data)
p.legend(["Finmag", "Nmag"])
p.figure()
p.plot(x, finmag_data)
p.legend(["Finmag"])
p.figure()
p.plot(x, nmag_data, label="Nmag")
p.legend(["Nmag"])
p.show()

