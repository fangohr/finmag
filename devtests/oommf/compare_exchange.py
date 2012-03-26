import os
import dolfin as df
import numpy as np
from finmag.sim.helpers import quiver, boxplot, finmag_to_oommf, stats

from finmag.sim.llg import LLG
from finmag.util.oommf import oommf_uniform_exchange, mesh

REL_TOLERANCE = 35 # goal: < 1e-3
MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

def test_one_dimensional_problem():
    results = one_dimensional_problem()
    assert np.nanmax(results["rel_diff"]) < REL_TOLERANCE

def test_three_dimensional_problem():
    results = three_dimensional_problem()
    assert np.nanmax(results["rel_diff"]) < REL_TOLERANCE

def compute_exc_finmag(mesh, m0, **kwargs):
    llg = LLG(mesh)
    llg.set_m0(m0, **kwargs)
    llg.setup()
    exc_field = df.Function(llg.V)
    exc_field.vector()[:] = llg.exchange.compute_field()
    return exc_field, llg

def one_dimensional_problem():
    x_min = 0; x_max = 20e-9; x_n = 40;
    
    # compute exchange field with finmag
    dolfmesh = df.Interval(x_n, x_min, x_max)
    m0_x = 'sqrt(x[0]/L)'
    # workaround floating point precision problems, getting really small
    # but negative values under the square root.
    m0_y_squared = '1 - x[0]/L - pow(sin(4*pi*x[0]/L)/10, 2)'
    m0_y = 'sqrt(fabs({0}) < 1e-15 ? 0: ({0}))'.format(m0_y_squared)
    m0_z = 'sin(4*pi*x[0]/L) / 10'
    exc_finmag, llg = compute_exc_finmag(dolfmesh, (m0_x, m0_y, m0_z), L=x_max)

    # compute exchange field with oommf
    msh = mesh.Mesh((x_n, 1, 1), size=(x_max, 1e-12, 1e-12))
    m0 = msh.new_field(3)
    for i, (x, y, z) in enumerate(msh.iter_coords()):
        m0.flat[0,i] = np.sqrt(x/x_max)
        m0.flat[1,i] = np.sqrt(1 - x/x_max
                - np.sin(4*np.pi*x/x_max)*np.sin(4*np.pi*x/x_max)/100)
        m0.flat[2,i] = np.sin(4 * np.pi * x/x_max)/10
    m0.flat /= np.sqrt(m0.flat[0]*m0.flat[0] +
                       m0.flat[1]*m0.flat[1] + m0.flat[2]*m0.flat[2])
    exchange_oommf = oommf_uniform_exchange(m0, llg.Ms, llg.C).flat

    # create data from finmag field for comparison with oommf
    exchange_for_oommf  = finmag_to_oommf(exc_finmag, msh, dims=1)

    difference = exchange_for_oommf.flat - exchange_oommf
    relative_difference = np.abs(difference / exchange_oommf)

    return dict(prob="1d", m0=llg.m, mesh=dolfmesh, oommf_mesh=msh,
            exc=exc_finmag.vector().array(), oommf_exc=exchange_oommf,
            diff=difference, rel_diff=relative_difference)

def three_dimensional_problem():
    x_max = 20e-9; y_max = 10e-9; z_max = 10e-9;
    dolfmesh = df.Box(0, 0, 0, x_max, y_max, z_max, 40, 10, 10)
    m0 = ("pow(sin(x[0] * pow(10, 9) / 3), 2)", "0", "1")
    exc_finmag, llg = compute_exc_finmag(dolfmesh, m0)

    msh = mesh.Mesh((20, 10, 10), size=(x_max, y_max, z_max))
    m0 = msh.new_field(3)
    for i, (x, y, z) in enumerate(msh.iter_coords()):
        m0.flat[0,i] = np.sin(10**9 * x/3)**2
        m0.flat[1,i] = 0
        m0.flat[2,i] = 1
    m0.flat /= np.sqrt(m0.flat[0]*m0.flat[0] + m0.flat[1]*m0.flat[1] + m0.flat[2]*m0.flat[2])
    exchange_oommf = oommf_uniform_exchange(m0, llg.Ms, llg.C).flat

    exchange_for_oommf = finmag_to_oommf(exc_finmag, msh, dims=3)

    difference = exchange_for_oommf.flat - exchange_oommf
    relative_difference = difference / exchange_oommf
    #relative_difference = np.abs(difference / np.sqrt(
    #    exchange_oommf[0]**2+exchange_oommf[1]**2+exchange_oommf[2]**2))

    return dict(prob="3d", m0=llg.m, mesh=dolfmesh, oommf_mesh=msh,
            exc=exc_finmag.vector().array(), oommf_exc=exchange_oommf,
            diff=difference, rel_diff=relative_difference)

if __name__ == '__main__':
    res1 = one_dimensional_problem()
    print "1D problem, relative difference:\n", stats(res1["rel_diff"])
    res3 = three_dimensional_problem()
    print "3D problem, relative difference:\n", stats(res3["rel_diff"])

    # Break it down. STOP! PLOTTER TIME.
    for res in [res1, res3]: 
        prefix = MODULE_DIR + res["prob"] + "_exc_"
        # images are worthless if the colormap is not shown. How to do that?
        quiver(res["m0"], res["mesh"], prefix+"m0.png", "1d m0")
        quiver(res["exc"], res["mesh"], prefix+"finmag.png", "1d finmag")
        quiver(res["oommf_exc"], res["oommf_mesh"], prefix+"oommf.png", "1d oommf")
        quiver(res["rel_diff"], res["oommf_mesh"], prefix+"rel_diff.png", "1d rel diff")
        boxplot(res["rel_diff"], prefix+"rel_diff_box.png")
