import dolfin as df
from finmag import Simulation
from finmag.energies import TimeZeeman, Demag, Exchange

"""
Micromag Standard Problem #4

specification:
    http://www.ctcms.nist.gov/~rdm/mumag.org.html

"""

L = 500e-9; W = 125e-9; H = 3e-9; # dimensions of film          nm
Ms = 8.0e5                        # saturation magnetisation    A/m
A = 1.3e-11                       # exchange coupling strength  J/m

mesh = df.Box(0, 0, 0, L, W, H, 166, 41, 1)
sim = Simulation(mesh, Ms)
sim.set_m((1, 1, 1))
sim.add(Demag())
sim.add(Exchange(A))

# not sure I need this if I initialise m with 1,1,1
saturating_field = df.Expression(("(tmax-t)*H","(tmax-t)*H","(tmax-t)*H"), tmax=1e-9, t=0, H=Ms)
H_ext = TimeZeeman(saturating_field)
sim.add(H_ext)

# interface for this is still rough
def update_H_ext(llg):
    H_ext.update(llg.t)
sim.llg._pre_rhs_callables.append(update_H_ext)

sim.run_until(2e-9)
df.plot(sim.llg._m)
df.interactive()

# reset t to 0
# apply one of the two fields
# let the system relax
#   record average magnetisation
#   when mx=0 for the first time, plot m
