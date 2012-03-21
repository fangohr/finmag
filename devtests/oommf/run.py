import dolfin as df
import numpy as np

from finmag.sim.llg import LLG
from finmag.sim.helpers import components
from finmag.util.oommf import oommf_uniform_exchange, mesh

"""
finmag code

"""
x0 = 0; y0 = z0 = 0
x1 = 10e-9; y1 = z1 = 1e-10
xn = 20; yn = zn = 1 
#msh = df.Interval(xn-1, x0, x1)
msh = df.Box(x0, y0, z0, x1, y1, z1, xn, yn, zn)

llg = LLG(msh)

# initial configuration of the magnetisation
m0_x = '2*x[0]/L - 1'
m0_y = 'sqrt(1 - (2*x[0]/L - 1)*(2*x[0]/L - 1))'
m0_z = '0'
llg.set_m0((m0_x, m0_y, m0_z), L=x1)
llg.setup(exchange_flag=True)

print components(llg.exchange.compute_field())

exc_finmag = df.Function(llg.V)
exc_finmag.vector()[:] = llg.exchange.compute_field()
df.plot(llg._m)
df.plot(exc_finmag)
df.interactive()

"""
oommf code

"""

print "\noommf\n"

msh = mesh.Mesh((xn, yn, zn), size=(x1, y1, z1))
m0 = msh.new_field(3)

for i, (x, y, z) in enumerate(msh.iter_coords()):
    m0.flat[0,i] = 2 * x/x1 - 1
    m0.flat[1,i] = np.sqrt(1 - (2*x/x1 - 1)*(2*x/x1 - 1))
    m0.flat[2,i] = 0

# m0.flat.shape == (3, n)
m0.flat /= np.sqrt(m0.flat[0]*m0.flat[0] + m0.flat[1]*m0.flat[1] + m0.flat[2]*m0.flat[2])

print oommf_uniform_exchange(m0, llg.Ms, llg.C).flat

