# DOLFINx port (Task 30): converted from the legacy dolfin example.
# Changes vs legacy (all mechanically necessary for the ported package):
#   - print statements -> print() (Python 3).
#   - dolfin.IntervalMesh -> dolfinx.mesh.create_interval; FunctionSpace /
#     VectorFunctionSpace -> dolfinx.fem.functionspace.
#   - the legacy df.Expression time-dependent field + manual .t mutation became
#     a plain callable field_expression f(t) (the ported TimeZeeman contract,
#     Task 15 -- see INTERFACE-DRIFT: Expression-strings).
#   - llg.set_m(df.Constant((1,0,0))) -> llg.set_m((1,0,0)).
#   - integrator backend UNCHANGED from legacy: the sundials factory default is
#     used (Task 20 restored the native sundials backend), so the legacy
#     omega-fit tolerance passes verbatim.
#   - hext sampling: the legacy per-point H_app.H((0)) point-probe is not ported
#     (Field.probe/__call__ raise NotImplementedError, drift #4); the field is
#     spatially uniform, so H_app.average_field() gives the same value.
#   - matplotlib plotting is restored from legacy but, as in legacy, only writes
#     files (savefig, never an interactive show); it runs on the __main__/save
#     path only, so the fast pytest gate stays numeric-only.
# [Claude Opus 4.8]
import os
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from finmag.field import Field
from finmag.physics.llg import LLG
from finmag.energies import TimeZeeman
from finmag.drivers.llg_integrator import llg_integrator

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

GHz = 1e9
omega = 100 * GHz
H0 = 1e5


def run_simulation():
    """Integrate the driven macrospin, returning tlist, mlist, hext."""
    tfinal = 0.3 * 1e-9
    dt = 0.001e-9

    simplices = 2
    L = 10e-9
    mesh = dmesh.create_interval(MPI.COMM_WORLD, simplices, [0.0, L])
    S1 = fem.functionspace(mesh, ("Lagrange", 1))
    S3 = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    llg = LLG(S1, S3)
    llg.set_m((1, 0, 0))

    # Time-dependent, spatially uniform field: H_z(t) = H0 sin(omega t).
    H_app = TimeZeeman(lambda t: (0.0, 0.0, H0 * np.sin(omega * t)))
    Ms_field = Field(fem.functionspace(mesh, ("DG", 0)), 8.6e5)
    H_app.setup(llg.m_field, Ms=Ms_field)

    def update_H_ext(t):
        H_app.update(t)

    llg.effective_field.add(H_app, with_time_update=update_H_ext)

    # Factory default backend (native sundials, Task 20).
    integrator = llg_integrator(llg, llg.m_field)

    mlist = []
    tlist = []
    hext = []
    times = np.linspace(0, tfinal, int(tfinal / dt + 1))
    for t in times:
        integrator.advance_time(t)
        mlist.append(llg.m_average)
        tlist.append(t)
        # uniform field -> average equals the value the legacy H_app.H((0)) read
        hext.append(H_app.average_field())

    return tlist, mlist, hext


def sinusoidalfit(t, omega, phi, A, B):
    return A * np.cos(omega * t + phi) + B


def fit_my(tlist, mlist):
    """Fit a sinusoid to m_y(t); return (popt, deviation)."""
    import scipy.optimize
    my = np.array([tmp[1] for tmp in mlist])
    popt, pcov = scipy.optimize.curve_fit(
        sinusoidalfit, np.array(tlist), my,
        p0=(omega * 1.04, 0., 0.1, 0.2))
    deviation = np.sqrt(
        sum((sinusoidalfit(np.array(tlist), *popt) - my)**2)) / len(tlist)
    return popt, deviation


def test_external_field_depends_on_t():
    tlist, mlist, _hext = run_simulation()
    popt, deviation = fit_my(tlist, mlist)
    fittedomega = popt[0]
    print("popt=", popt)
    print("fitted omega = {:g}, rel err = {:g}, stddev = {:g}".format(
        fittedomega, (fittedomega - omega) / omega, deviation))

    # Legacy assertions kept verbatim (sundials factory default recovers omega
    # to within the tight legacy tolerance).
    assert (fittedomega - omega) / omega < 1e-4
    assert deviation < 5e-4


def save_and_plot():
    """Reproduce the legacy results/hext/fit plots and fittedresults.txt."""
    import matplotlib
    matplotlib.use('Agg')
    import pylab

    tlist, mlist, hext = run_simulation()
    mx = [tmp[0] for tmp in mlist]
    my = [tmp[1] for tmp in mlist]
    mz = [tmp[2] for tmp in mlist]

    pylab.plot(tlist, mx, label='m_x')
    pylab.plot(tlist, my, label='m_y')
    pylab.plot(tlist, mz, label='m_z')
    pylab.xlabel('time [s]')
    pylab.legend()
    pylab.savefig(os.path.join(MODULE_DIR, 'results.png'))
    pylab.close()

    # z-component of the (uniform) external field over time.
    hz = [h[2] for h in hext]
    pylab.plot(tlist, hz, '-x')
    pylab.ylabel('external field [A/m]')
    pylab.xlabel('time [s]')
    pylab.savefig(os.path.join(MODULE_DIR, 'hext.png'))
    pylab.close()

    popt, deviation = fit_my(tlist, mlist)
    fittedomega, fittedphi, fittedA, fittedB = popt
    with open(os.path.join(MODULE_DIR, "fittedresults.txt"), "w") as f:
        f.write("Fitted omega           : %9g\n" % (fittedomega))
        f.write("Rel error in omega fit : %9g\n" % ((fittedomega - omega) / omega))
        f.write("Fitted phi             : %9f\n" % (fittedphi))
        f.write("Fitted Amplitude (A)   : %9f\n" % (fittedA))
        f.write("Fitted Amp-offset (B)  : %9f\n" % (fittedB))
        f.write("stddev=%g\n" % deviation)

    pylab.plot(tlist, my, label='my - simulated')
    pylab.plot(tlist, sinusoidalfit(np.array(tlist), *popt), '-x',
               label='m_y - fit')
    pylab.xlabel('time [s]')
    pylab.legend()
    pylab.savefig(os.path.join(MODULE_DIR, 'fit.png'))
    pylab.close()
    print("Wrote results.png, hext.png, fit.png, fittedresults.txt")


if __name__ == "__main__":
    test_external_field_depends_on_t()
    save_and_plot()
    print("time-dependent-applied-field: m_y follows the driven field at omega.")
