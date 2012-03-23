import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from finmag.sim.llg import LLG
from finmag.util.oommf import oommf_uniform_exchange, mesh

REL_TOLERANCE = 40 # 1e-3

def compute_exchange_finmag(mesh, m0, **kwargs):
    llg = LLG(mesh)
    llg.set_m0(m0, **kwargs)
    llg.setup()
    exc_field = df.Function(llg.V)
    exc_field.vector()[:] = llg.exchange.compute_field()
    return exc_field, llg.Ms, llg.C

def finmag_to_oommf(finmag_exchange, oommf_mesh, dims=1):
    exchange_finmag_for_oommf = oommf_mesh.new_field(3)
    for i, (x, y, z) in enumerate(oommf_mesh.iter_coords()):
        if dims == 1:
            E_x, E_y, E_z = finmag_exchange(x)
        else:
            E_x, E_y, E_z = finmag_exchange(x, y, z)
        exchange_finmag_for_oommf.flat[0,i] = E_x
        exchange_finmag_for_oommf.flat[1,i] = E_y
        exchange_finmag_for_oommf.flat[2,i] = E_z
    return exchange_finmag_for_oommf

def test_one_dimensional_problem():
    x_min = 0; x_max = 20e-9; x_n = 20
    
    # compute exchange field with finmag
    msh = df.Interval(x_n, x_min, x_max)
    m0 = ('sqrt(x[0]/L)','sqrt(1 - sqrt(x[0]/L)*sqrt(x[0]/L))', '0')
    exchange_finmag, Ms, C = compute_exchange_finmag(msh, m0, L=x_max)

    # compute exchange field with oommf
    msh = mesh.Mesh((x_n, 1, 1), size=(x_max, 1e-11, 1e-11))
    m0 = msh.new_field(3)
    for i, (x, y, z) in enumerate(msh.iter_coords()):
        m0.flat[0,i] = np.sqrt(x/x_max)
        m0.flat[1,i] = np.sqrt(1 - np.sqrt(x/x_max)*np.sqrt(x/x_max))
        m0.flat[2,i] = 0
    m0.flat /= np.sqrt(m0.flat[0]*m0.flat[0] +
                       m0.flat[1]*m0.flat[1] + m0.flat[2]*m0.flat[2])
    exchange_oommf = oommf_uniform_exchange(m0, Ms, C).flat

    # create data from finmag field for comparison with oommf
    exchange_for_oommf  = finmag_to_oommf(exchange_finmag, msh, dims=1)

    difference = exchange_for_oommf.flat - exchange_oommf
    relative_difference = np.abs(difference / exchange_oommf)
    assert np.nanmax(relative_difference) < REL_TOLERANCE

    return difference, relative_difference

def test_three_dimensional_problem():
    x_max = 20e-9; y_max = 10e-9; z_max = 10e-9;
    msh = df.Box(0, 0, 0, x_max, y_max, z_max, 40, 20, 20)
    m0 = ("pow(sin(x[0] * pow(10, 9) / 3), 2)", "0", "1")
    exchange_finmag, Ms, C = compute_exchange_finmag(msh, m0)

    msh = mesh.Mesh((20, 10, 10), size=(x_max, y_max, z_max))
    m0 = msh.new_field(3)
    for i, (x, y, z) in enumerate(msh.iter_coords()):
        m0.flat[0,i] = np.sin(10**9 * x/3)**2
        m0.flat[1,i] = 0
        m0.flat[2,i] = 1
    m0.flat /= np.sqrt(m0.flat[0]*m0.flat[0] + m0.flat[1]*m0.flat[1] + m0.flat[2]*m0.flat[2])
    exchange_oommf = oommf_uniform_exchange(m0, Ms, C).flat

    exchange_for_oommf = finmag_to_oommf(exchange_finmag, msh, dims=3)
    difference = exchange_for_oommf.flat - exchange_oommf
    relative_difference = np.abs(difference / exchange_oommf)
    assert np.nanmax(relative_difference) < REL_TOLERANCE

    return difference, relative_difference

def stats_from_array_str(arr):
    median  = np.median(arr)
    average = np.mean(arr, axis=1)
    minimum = np.nanmin(arr)
    maximum = np.nanmax(arr)
    spread  = np.std(arr, axis=1)
    stats= "  min, median, max = ({0}, {1} {2}),\n  means = {3}),\n  stds = {4}".format(
            minimum, median, maximum, average, spread)
    return stats

def boxplot(arr, filename):
    plt.boxplot(arr)
    plt.show()

def show_quiverplot(msh, diff):
    x, y, z = zip(* msh.iter_coords())
    figure = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    q = mlab.quiver3d(x, y, z, diff[0], diff[1], diff[2],
            figure=figure, mode="cone", colormap="hot")
    q.scene.z_plus_view()
    mlab.axes(figure=figure)
    mlab.show()

if __name__ == '__main__':
    diff_1d, rel_diff_1d = test_one_dimensional_problem()
    print "1D problem, relative difference:\n", stats_from_array_str(rel_diff_1d)

    diff_3d, rel_diff_3d = test_three_dimensional_problem()
    print "3D problem, relative difference:\n", stats_from_array_str(rel_diff_3d)
