import os
import dolfin as df
from scipy.integrate import ode
from finmag.sim.llg import LLG

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_simulation():
    """
    Translation of the nmag code.
    Mesh generated on the fly.

    """

    x0 = 0; x1 = 15e-9; nx = 30;
    y0 = -4.5e-9; y1 = 4.5e-9; ny = 18;
    z0 = -0.1e-9; z1 = 0.1e-9; nz = 1;
    mesh = df.Box(x0, y0, z0, x1, y1, z1, nx, ny, nz) 
    nb_nodes = len(mesh.coordinates())

    llg = LLG(mesh)
    llg.Ms = 1e6
    llg.A = 1.3e-11
    #llg.c = 1e11
    llg.alpha = 0.02

    llg.set_m0(("1",
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
    llg.setup(use_exchange=True)

    llg_wrap = lambda t, y: llg.solve_for(y, t)
    t0 = 0; dt = 0.05e-12; t1 = 10e-12
    r = ode(llg_wrap).set_integrator("vode", method="bdf", rtol=1e-5, atol=1e-5)
    r.set_initial_value(llg.m, t0)

    fh = open(MODULE_DIR + "/averages.txt", "w")
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
