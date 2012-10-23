import dolfin
import os
import numpy

from finmag.sim.integrator import LLGIntegrator
from finmag.sim.llg import LLG
from finmag.energies import Zeeman
from finmag.util.macrospin import make_analytic_solution

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

"""
The analytical solution of the LLG equation for a constant
applied field, based on Appendix B of Matteo's PhD thesis,
pages 127-128, equations B.16-B.18.

"""

def compare_with_analytic_solution(alpha=0.5, max_t=1e-9):
    """
    Compares the C/dolfin/odeint solution to the analytical one.

    """
    print "Running comparison with alpha={0}.".format(alpha)

    # define 3d mesh
    x0 = y0 = z0 = 0
    x1 = y1 = z1 = 10e-9
    nx = ny = nz = 1
    mesh = dolfin.Box(x0, x1, y0, y1, z0, z1, nx, ny, nz)
    S1 = dolfin.FunctionSpace(mesh, "Lagrange", 1)
    S3 = dolfin.VectorFunctionSpace(mesh, "Lagrange", 1)

    llg = LLG(S1, S3)
    llg.alpha = alpha
    llg.set_m((1, 0, 0))
    H_app = Zeeman((0, 0, 1e6))
    H_app.setup(S3, llg._m, Ms=1)
    llg.effective_field.add(H_app)

    integrator = LLGIntegrator(llg, llg.m, abstol=1e-12, reltol=1e-12)
    ts = numpy.linspace(0, max_t, num=100)
    tsfine = numpy.linspace(0, max_t, num=1000)
    ys = numpy.array([(integrator.run_until(t), integrator.m.copy())[1] for t in ts])
    m_analytical = make_analytic_solution(1e6, alpha, llg.gamma)
    save_plot(ts, ys, tsfine, m_analytical, alpha)

    TOLERANCE = 1e-6  # tolerance on Ubuntu 11.10, VM Hans, 25/02/2012

    rel_diff_maxs = list()
    for i in range(len(ts)):
        m = numpy.mean(ys[i].reshape((3, -1)), axis=1)
        m_ref = m_analytical(ts[i])
        diff = numpy.abs(m - m_ref)
        diff_max = numpy.max(diff)
        rel_diff_max = numpy.max(diff / numpy.max(m_ref))
        rel_diff_maxs.append(rel_diff_max)

        print "t= {0:.3g}, diff_max= {1:.3g}.".format(ts[i], diff_max)

        msg = "Diff at t= {0:.3g} too large.\nAllowed {1:.3g}. Got {2:.3g}."
        assert diff_max < TOLERANCE, msg.format(ts[i], TOLERANCE, diff_max)
    print "Maximal relative difference: "
    print numpy.max(numpy.array(rel_diff_maxs))


def save_plot(ts, ys, ts_ref, m_ref, alpha):
    ys3d = ys.reshape((len(ys), 3, 8)).mean(axis=-1)
    mx = ys3d[:,0]
    my = ys3d[:,1]
    mz = ys3d[:,2]
    print "mx.shape", mx.shape
    print "m_analytical.shape", m_ref(ts).shape
    m_exact = m_ref(ts_ref)
    mx_exact = m_exact[0,:]
    my_exact = m_exact[1,:]
    mz_exact = m_exact[2,:]

    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    plt.plot(ts, mx, 'o', label='mx')
    plt.plot(ts, my, 'x', label='my')
    plt.plot(ts, mz, '^', label='mz')
    plt.plot(ts_ref, mx_exact, '-', label='mx (exact)')
    plt.plot(ts_ref, my_exact, '-', label='my (exact)')
    plt.plot(ts_ref, mz_exact, '-', label='mz (exact)')
    plt.xlabel('t [s]')
    plt.ylabel('m=M/Ms')
    plt.title(r'Macrospin dynamics: $\alpha$={}'.format(alpha))
    plt.grid()
    plt.legend()
    filename = ('alpha-%04.2f' % alpha)
    #latex does not like multiple '.' in image filenames
    filename = filename.replace('.', '-')
    plt.savefig(os.path.join(MODULE_DIR, filename + '.pdf'))
    plt.savefig(os.path.join(MODULE_DIR, filename + '.png'))
    plt.close()
    #pylab.show()


def test_macrospin_very_low_damping():
    compare_with_analytic_solution(alpha=0.02, max_t=2e-9)


def test_macrospin_low_damping():
    compare_with_analytic_solution(alpha=0.1, max_t=4e-10)


def test_macrospin_standard_damping():
    compare_with_analytic_solution(alpha=0.5, max_t=1e-10)


def test_macrospin_higher_damping():
    compare_with_analytic_solution(alpha=1, max_t=1e-10)


if __name__ == "__main__":
    test_macrospin_very_low_damping()
    test_macrospin_low_damping()
    test_macrospin_standard_damping()
    test_macrospin_higher_damping()
