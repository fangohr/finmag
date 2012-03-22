import dolfin as df
import numpy as np
from mayavi import mlab
from finmag.sim.llg import LLG
from finmag.util.oommf import oommf_uniaxial_anisotropy, mesh

L = 20e-9; W = 10e-9; H = 1e-9;
nL = 20; nW = 10; nH = 1;

K1 = 45e4 # J/m^3

def one_dimensional_problem():
    msh = df.Interval(nL, 0, L)
    llg = LLG(msh)
    m0_x = '2 * x[0]/L - 1'
    m0_y = 'sqrt(1 - (2*x[0]/L - 1)*(2*x[0]/L - 1))'
    m0_z = '0'
    llg.set_m0((m0_x, m0_y, m0_z), L=L)
    llg.add_uniaxial_anisotropy(K1, df.Constant((0, 0, 1)))
    llg.setup(exchange_flag=True)
    return llg

def three_dimensional_problem():
    msh = df.Box(0, 0, 0, L, W, H, nL, nW, nH)
    llg = LLG(msh)
    m0_x = "pow(sin(x[0]*pow(10, 9)/3), 2)"
    m0_y = "0"
    m0_z = "1" # "pow(cos(x[0]*pow(10, 9)/3), 2)" 
    llg.set_m0((m0_x, m0_y, m0_z))
    llg.add_uniaxial_anisotropy(K1, df.Constant((1, 0, 0)))
    llg.setup(exchange_flag=True)
    return llg

#llg = one_dimensional_problem()
llg = three_dimensional_problem()

anis_finmag = df.Function(llg.V)
# There is only one anisotropy in our example.
anis_finmag.vector()[:] = llg._anisotropies[0].compute_field()

def one_dimensional_problem_oommf():
    msh = mesh.Mesh((nL, 1, 1), size=(L, 1e-10, 1e-10))
    m0 = msh.new_field(3)

    for i, (x, y, z) in enumerate(msh.iter_coords()):
        m0.flat[0,i] = 2 * x/L - 1
        m0.flat[1,i] = np.sqrt(1 - (2*x/L - 1)*(2*x/L - 1))
        m0.flat[2,i] = 0

    # m0.flat.shape == (3, n)
    m0.flat /= np.sqrt(m0.flat[0]*m0.flat[0] + m0.flat[1]*m0.flat[1] + m0.flat[2]*m0.flat[2])

    return msh, oommf_uniaxial_anisotropy(m0, llg.Ms, K1, (0,0,1)).flat

def three_dimensional_problem_oommf():
    msh = mesh.Mesh((nL, nW, nH), size=(L, W, H))
    m0 = msh.new_field(3)

    for i, (x, y, z) in enumerate(msh.iter_coords()):
        m0.flat[0,i] = np.sin(10**9 * x/3)**2
        m0.flat[1,i] = 0
        m0.flat[2,i] = 1 # np.cos(10**9 * x/3)

    # m0.flat.shape == (3, n)
    m0.flat /= np.sqrt(m0.flat[0]*m0.flat[0] + m0.flat[1]*m0.flat[1] + m0.flat[2]*m0.flat[2])

    return msh, oommf_uniaxial_anisotropy(m0, llg.Ms, K1, (1,0,0)).flat

#msh, anis_oommf = one_dimensional_problem_oommf()
msh, anis_oommf = three_dimensional_problem_oommf()

anis_finmag_like_oommf = msh.new_field(3)
for i, (x, y, z) in enumerate(msh.iter_coords()):
    # A_x, A_y, A_z = anis_finmag(x) # one dimension
    A_x, A_y, A_z = anis_finmag(x, y, z)
    anis_finmag_like_oommf.flat[0,i] = A_x
    anis_finmag_like_oommf.flat[1,i] = A_y
    anis_finmag_like_oommf.flat[2,i] = A_z

difference = anis_finmag_like_oommf.flat - anis_oommf
relative_difference = np.abs(difference / anis_oommf)

print "difference (x-component)"
print difference[0]
print "absolute relative difference (x-component)"
print relative_difference[0]
print "absolute relative difference (x-component):"
print "  median", np.median(relative_difference[0])
print "  average", np.mean(relative_difference[0])
print "  minimum", np.min(relative_difference[0])
print "  maximum", np.max(relative_difference[0])
print "  spread", np.std(relative_difference[0])

x, y, z = zip(* msh.iter_coords())
figure = mlab.figure(bgcolor=(0, 0, 0), fgcolor=(1, 1, 1))
q = mlab.quiver3d(x, y, z, difference[0], difference[1], difference[2], figure=figure)
q.scene.z_plus_view()
mlab.axes(figure=figure)
mlab.show()
