import dolfin as df
import numpy as np

from finmag.sim.llg import LLG
from finmag.util.oommf import oommf_dmdt, mesh

# disable the computation of the exchange and anisotropy fields
# to have the same effective field for the comparison of the
# results for dm/dt. What we do have is an external field.

L = 20e-9; W = 10e-9; H = 1e-9;
nL = 20; nW = 10; nH = 1;

mh = df.Box(0, 0, 0, L, W, H, nL, nW, nH)
llg = LLG(mh)
m0_x = "-3"
m0_y = "-2"
m0_z = "1"
llg.set_m0((m0_x, m0_y, m0_z))

h = llg.Ms/2
H_app = (h/np.sqrt(3), h/np.sqrt(3), h/np.sqrt(3))
llg.H_app = H_app

llg.setup(exchange_flag=False)

dmdt_finmag = df.Function(llg.V)
dmdt_finmag.vector()[:] = llg.solve()

df.plot(llg._m, title="m")
df.plot(llg._H_app, title="H_app")
df.plot(dmdt_finmag, title="dmdt")
df.interactive()

msh = mesh.Mesh((nL, nW, nH), size=(L, W, H))
m0 = msh.new_field(3)

for i, (x, y, z) in enumerate(msh.iter_coords()):
    m0.flat[0,i] = -3 # we don't need to loop for such constant values
    m0.flat[1,i] = -2 # but having this code in place might prove useful
    m0.flat[2,i] = 1 

# m0.flat.shape == (3, n)
m0.flat /= np.sqrt(m0.flat[0]*m0.flat[0] + m0.flat[1]*m0.flat[1] + m0.flat[2]*m0.flat[2])

dmdt_oommf = oommf_dmdt(m0, llg.Ms, A=llg.C, H=H_app, alpha=llg.alpha, gamma_G=llg.gamma).flat

dmdt_finmag_like_oommf = msh.new_field(3)
for i, (x, y, z) in enumerate(msh.iter_coords()):
    dmdt_x, dmdt_y, dmdt_z = dmdt_finmag(x, y, z)
    dmdt_finmag_like_oommf.flat[0,i] = dmdt_x
    dmdt_finmag_like_oommf.flat[1,i] = dmdt_y
    dmdt_finmag_like_oommf.flat[2,i] = dmdt_z

difference = dmdt_finmag_like_oommf.flat - dmdt_oommf
relative_difference = difference / dmdt_oommf
print "difference"
print difference
print "relative difference"
print relative_difference


