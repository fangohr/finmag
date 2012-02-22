import dolfin as df
import numpy as np
import finmag.sim.helpers as h
import time

from scipy.integrate import ode
from finmag.sim.llg import LLG

fh = open("data_M_dolfin.txt", "w")

mesh = df.Mesh("bar.xml")
llg = LLG(mesh)

llg.alpha = 0.1
llg.MS = 0.86e6 # A/m
llg.C = 1.3e-11 # J/m
llg.H_app = (0.43e6, 0, 0) # A/m

# non-uniform starting magnetisation
nodes = mesh.num_vertices()
vectors = h.perturbed_vectors(nodes, [1,1,0], llg.MS)
print [h.norm(v) for v in vectors]
vectors_for_dolfin = h.rows_to_columns(vectors).flatten()
llg._M = df.Function(llg.V) # no API for initialising with pre-computed field.
llg.M = vectors_for_dolfin

llg.setup(exchange_flag=False)

t0 = 0; dt = 1e-12; tmax = 1e-9 # s
llg_wrap = lambda t, y: llg.solve_for(y, t)
r = ode(llg_wrap).set_integrator("vode", method="bdf", with_jacobian=False)
r.set_initial_value(llg.M, t0)
while r.successful() and r.t <= tmax:
    df.plot(llg._M)
    Mx, My, Mz = llg.average_M()
    fh.write(str(r.t) + " " + str(Mx) + " " + str(My) + " " + str(Mz) + "\n")
    r.integrate(r.t+dt)
fh.close()
df.interactive()
