import dolfin as df
import numpy as np

from finmag.sim.llg import LLG
from finmag.util.oommf import oommf_uniform_exchange, mesh

L = 20e-9; W = 10e-9; H = 1e-9;
nL = 40; nW = 10; nH = 1;

def one_dimensional_problem():
    msh = df.Interval(nL, 0, L)
    #msh = df.Box(0, 0, 0, L, 1e-10, 1e-10, nL, 1, 1)
    llg = LLG(msh)
    m0_x = 'sqrt(x[0]/L)'
    m0_y = 'sqrt(1 - sqrt(x[0]/L)*sqrt(x[0]/L))'
    m0_z = '0'
    llg.set_m0((m0_x, m0_y, m0_z), L=L)
    llg.setup(exchange_flag=True)
    return llg

def three_dimensional_problem():
    msh = df.Box(0, 0, 0, L, W, H, nL, nW, nH)
    llg = LLG(msh)
    m0_x = "pow(sin(x[0]*pow(10, 9)/3), 2)"
    m0_y = "0"
    m0_z = "1" # "pow(cos(x[0]*pow(10, 9)/3), 2)" 
    llg.set_m0((m0_x, m0_y, m0_z))
    llg.setup(exchange_flag=True)
    return llg

llg = one_dimensional_problem()
#llg = three_dimensional_problem()

exc_finmag = df.Function(llg.V)
exc_finmag.vector()[:] = llg.exchange.compute_field()

#df.plot(llg._m)
#df.plot(exc_finmag)
#df.interactive()

def one_dimensional_problem_oommf():
    msh = mesh.Mesh((nL, 1, 1), size=(L, 1e-11, 1e-11))
    m0 = msh.new_field(3)

    for i, (x, y, z) in enumerate(msh.iter_coords()):
        m0.flat[0,i] = np.sqrt(x/L)
        m0.flat[1,i] = np.sqrt(1 - np.sqrt(x/L)*np.sqrt(x/L))
        m0.flat[2,i] = 0

    # m0.flat.shape == (3, n)
    m0.flat /= np.sqrt(m0.flat[0]*m0.flat[0] + m0.flat[1]*m0.flat[1] + m0.flat[2]*m0.flat[2])

    return msh, oommf_uniform_exchange(m0, llg.Ms, llg.C).flat

def three_dimensional_problem_oommf():
    msh = mesh.Mesh((nL, nW, nH), size=(L, W, H))
    m0 = msh.new_field(3)

    for i, (x, y, z) in enumerate(msh.iter_coords()):
        m0.flat[0,i] = np.sin(10**9 * x/3)**2
        m0.flat[1,i] = 0
        m0.flat[2,i] = 1 # np.cos(10**9 * x/3)

    # m0.flat.shape == (3, n)
    m0.flat /= np.sqrt(m0.flat[0]*m0.flat[0] + m0.flat[1]*m0.flat[1] + m0.flat[2]*m0.flat[2])

    return msh, oommf_uniform_exchange(m0, llg.Ms, llg.C).flat

msh, exc_oommf = one_dimensional_problem_oommf()
#msh, exc_oommf = three_dimensional_problem_oommf()

exc_finmag_like_oommf = msh.new_field(3)
for i, (x, y, z) in enumerate(msh.iter_coords()):
    E_x, E_y, E_z = exc_finmag(x, y, z) # one dimension
    #E_x, E_y, E_z = exc_finmag(x, y, z)
    exc_finmag_like_oommf.flat[0,i] = E_x
    exc_finmag_like_oommf.flat[1,i] = E_y
    exc_finmag_like_oommf.flat[2,i] = E_z

difference = exc_finmag_like_oommf.flat - exc_oommf
relative_difference = np.abs(difference / exc_oommf)

print "difference"
print difference
print "absolute relative difference"
print relative_difference
print "absolute relative difference:"
print "  median", np.median(relative_difference, axis=1)
print "  average", np.mean(relative_difference, axis=1)
print "  minimum", np.min(relative_difference, axis=1)
print "  maximum", np.max(relative_difference, axis=1)
print "  spread", np.std(relative_difference, axis=1)
