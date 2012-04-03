import os
import dolfin as df
import numpy as np
from finmag.sim.llg import LLG
from finmag.util.oommf import oommf_uniaxial_anisotropy, mesh
from finmag.sim.helpers import quiver, boxplot, stats
from finmag.util.oommf.comparison import finmag_to_oommf

K1 = 45e4 # J/m^3

REL_TOLERANCE = 2e-1 # goal: < 1e-3
MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

def test_small_problem():
    results = small_problem()
    assert np.nanmax(results["rel_diff"]) < REL_TOLERANCE

def test_one_dimensional_problem():
    results = one_dimensional_problem()
    assert np.nanmax(results["rel_diff"]) < REL_TOLERANCE

def test_three_dimensional_problem():
    results = three_dimensional_problem()
    assert np.nanmax(results["rel_diff"]) < REL_TOLERANCE

def compute_anis_finmag(mesh, m0, **kwargs):
    llg = LLG(mesh)
    llg.set_m0(m0, **kwargs)
    invsq3 = 1/np.sqrt(3)
    llg.add_uniaxial_anisotropy(K1, df.Constant((invsq3, invsq3, invsq3)))
    llg.setup()
    anis_field = df.Function(llg.V)
    anis_field.vector()[:] = llg._anisotropies[0].compute_field()
    return anis_field, llg

def small_problem():
    # The oommf mesh corresponding to this problem only has a single cell.
    x_max = 1e-9; y_max = 1e-9; z_max = 1e-9;
    dolfmesh = df.Box(0, 0, 0, x_max, y_max, z_max, 5, 5, 5)
    anis_finmag, llg = compute_anis_finmag(dolfmesh, (1, 0, 0))

    msh = mesh.Mesh((1, 1, 1), size=(x_max, y_max, z_max))
    m0 = msh.new_field(3)
    m0.flat[0] = np.ones(len(m0.flat[0]))
    anis_oommf = oommf_uniaxial_anisotropy(m0, llg.Ms, K1, (1,1,1)).flat
    anis_for_oommf = finmag_to_oommf(anis_finmag, msh, dims=3).flat

    difference = np.abs(anis_for_oommf - anis_oommf)
    relative_difference = difference / np.sqrt(
        anis_oommf[0]**2 + anis_oommf[1]**2 + anis_oommf[2]**2)

    return dict(prob="single", m0=llg.m, mesh=dolfmesh, oommf_mesh=msh,
            finmag=anis_finmag.vector().array(),
            oommf=anis_oommf,
            foommf=anis_for_oommf,
            diff=difference, rel_diff=relative_difference)

def one_dimensional_problem():
    x_min = 0; x_max = 10e-9; x_n = 40;
    dolfmesh = df.Interval(x_n, x_min, x_max)
    m0_x = '2 * x[0]/L - 1'
    m0_y = 'sqrt(1 - (2*x[0]/L - 1)*(2*x[0]/L - 1))'
    m0_z = '0'
    anis_finmag, llg = compute_anis_finmag(dolfmesh, (m0_x, m0_y, m0_z), L=x_max)

    msh = mesh.Mesh((x_n, 1, 1), size=(x_max, 1e-12, 1e-12))
    m0 = msh.new_field(3)
    for i, (x, y, z) in enumerate(msh.iter_coords()):
        m0.flat[0,i] = 2 * x/x_max - 1
        m0.flat[1,i] = np.sqrt(1 - (2*x/x_max - 1)*(2*x/x_max - 1))
        m0.flat[2,i] = 0
    m0.flat /= np.sqrt(m0.flat[0]*m0.flat[0] + m0.flat[1]*m0.flat[1] + m0.flat[2]*m0.flat[2])
    anis_oommf = oommf_uniaxial_anisotropy(m0, llg.Ms, K1, (1,1,1)).flat
    anis_for_oommf = finmag_to_oommf(anis_finmag, msh, dims=1)

    difference = np.abs(anis_for_oommf - anis_oommf)
    relative_difference = difference / np.sqrt(
        anis_oommf[0]**2 + anis_oommf[1]**2 + anis_oommf[2]**2)

    return dict(prob="1d", m0=llg.m, mesh=dolfmesh, oommf_mesh=msh,
            finmag=anis_finmag.vector().array(),
            foommf=anis_for_oommf,
            oommf=anis_oommf,
            diff=difference, rel_diff=relative_difference)

def three_dimensional_problem():
    x_max = 20e-9; y_max = 10e-9; z_max = 10e-9;
    dolfmesh = df.Box(0, 0, 0, x_max, y_max, z_max, 40, 20, 20)
    m0_x = "pow(sin(x[0]*pow(10, 9)/3), 2)"
    m0_y = "0"
    m0_z = "1"
    anis_finmag, llg = compute_anis_finmag(dolfmesh, (m0_x, m0_y, m0_z))

    msh = mesh.Mesh((20, 10, 10), size=(x_max, y_max, z_max))
    m0 = msh.new_field(3)
    for i, (x, y, z) in enumerate(msh.iter_coords()):
        m0.flat[0,i] = np.sin(10**9 * x/3)**2
        m0.flat[1,i] = 0
        m0.flat[2,i] = 1
    m0.flat /= np.sqrt(m0.flat[0]*m0.flat[0] + m0.flat[1]*m0.flat[1] + m0.flat[2]*m0.flat[2])
    anis_oommf = oommf_uniaxial_anisotropy(m0, llg.Ms, K1, (1,1,1)).flat
    anis_for_oommf = finmag_to_oommf(anis_finmag, msh, dims=3).flat

    difference = anis_for_oommf - anis_oommf
    relative_difference = np.abs(difference / anis_oommf)

    return dict(prob="3d", m0=llg.m, mesh=dolfmesh, oommf_mesh=msh,
            finmag=anis_finmag.vector().array(),
            foommf=anis_for_oommf,
            oommf=anis_oommf,
            diff=difference, rel_diff=relative_difference)

if __name__ == '__main__':

    res0 = small_problem()
    print "0D problem, relative difference:\n", stats(res0["rel_diff"])
    res1 = one_dimensional_problem()
    print "1D problem, relative difference:\n", stats(res1["rel_diff"])
    """
    res3 = three_dimensional_problem()
    print "3D problem, relative difference:\n", stats(res3["rel_diff"])
    """
