import dolfin as df
import numpy as np

from finmag.sim.llg import LLG
from finmag.sim.helpers import components
from finmag.util.oommf import oommf_uniform_exchange, mesh

"""
finmag code

"""
x0 = y0 = z0 = 0
x1 = y1 = z1 = 10e-9
xn = yn = zn = 10 
msh = df.Box(x0, y0, z0, x1, y1, z1, xn, yn, zn)

llg = LLG(msh)
llg.set_m0(("2*x[0]/L - 1", "2*x[1]/W - 1", "1"), L=x1, W=y1)
llg.H_app = (llg.Ms/2, 0, 0)
llg.setup(exchange_flag=True)

print "finmag\n\nm0"
print llg.m
print "\nexchange:"
print components(llg.exchange.compute_field())

"""
oommf code

"""

print "\noommf\n\n"

msh = mesh.Mesh((xn, yn, zn), size=(x1, y1, z1))
m0 = msh.new_field(3)

for i, (x, y, z) in enumerate(msh.iter_coords()):
    m0.flat[0,i] = 2 * x/x1 - 1
    m0.flat[1,i] = 2 * y/y1 -1
    m0.flat[2,i] = 1

# m0.flat.shape == (3, n)
m0.flat /= np.sqrt(m0.flat[0]*m0.flat[0] + m0.flat[1]*m0.flat[1] + m0.flat[2]*m0.flat[2])

print oommf_uniform_exchange(m0, llg.Ms, llg.C).flat
