import os
from scipy.integrate import ode
from finmag.sim.llg import LLG
from finmag.energies import Exchange
import dolfin as df

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_simulation():
    """
    Translation of the nmag code.
    Mesh generated on the fly.

    """

    x0 = 0; x1 = 15e-9; nx = 30;
    y0 = -4.5e-9; y1 = 4.5e-9; ny = 18;
    z0 = -0.1e-9; z1 = 0.1e-9; nz = 1;
    mesh = df.BoxMesh(x0, y0, z0, x1, y1, z1, nx, ny, nz)
    S1 = df.FunctionSpace(mesh, "Lagrange", 1)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)

    nb_nodes = len(mesh.coordinates())

    llg = LLG(S1, S3)
    llg.Ms = 1e6
    llg.set_alpha(0.02)

    exchange = Exchange(1.3e-11)
    exchange.setup(S3, llg._m, llg.Ms)
    llg.effective_field.add(exchange)

    llg.set_m(("1",
        "5 * pow(cos(pi * (x[0] * pow(10, 9) - 11) / 6), 3) \
           * pow(cos(pi * x[1] * pow(10, 9) / 6), 3)",
        "0"))

    m = llg.m
    for i in xrange(nb_nodes):
        x, y, z = mesh.coordinates()[i]
        mx = 1; my = 0; mz = 0;
        if 8e-9 < x < 14e-9 and -3e-9 < y < 3e-9:
            pass
        else:
            m[i] = mx; m[i+nb_nodes] = my; m[i+2*nb_nodes] = mz;
    llg.m = m

    llg_wrap = lambda t, y: llg.solve_for(y, t)
    t0 = 0; dt = 0.05e-12; t1 = 10e-12
    r = ode(llg_wrap).set_integrator("vode", method="bdf", rtol=1e-5, atol=1e-5)
    r.set_initial_value(llg.m, t0)

    fh = open(os.path.join(MODULE_DIR, "averages.txt"), "w")
    while r.successful() and r.t <= t1:
        print "Integrating time = %gs" % (r.t)
        mx, my, mz = llg.m_average
        fh.write(str(r.t) + " " + str(mx) + " " + str(my) + " " + str(mz) + "\n")
        r.integrate(r.t + dt)
        #df.plot(llg._m)
    fh.close()
    print "Done"
    #df.interactive()

if __name__ == "__main__":
    run_simulation()
