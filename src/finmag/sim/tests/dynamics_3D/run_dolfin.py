import os
import subprocess
import dolfin as df
import numpy as np
import finmag.sim.helpers as h
from scipy.integrate import ode
from finmag.sim.llg import LLG

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_simulation():
    """
    Create the mesh.

    """
    geofile      = MODULE_DIR + "/bar.geo"
    intermediate = MODULE_DIR + "/bar.gmsh"
    meshfile     = MODULE_DIR + "/bar.xml"

    create_command = 'NETGENDIR=/usr/share/netgen netgen -geofile="{0}" -meshfiletype="Gmsh2 Format" -meshfile="{1}" -batchmode'.format(geofile, intermediate)
    convert_command = 'dolfin-convert "{0}" "{1}"'.format(intermediate, meshfile)

    subprocess.call(create_command, shell=True)
    subprocess.call(convert_command, shell=True)

    mesh = df.Mesh(meshfile)
    mesh.coordinates()[:] = 1e-9 * mesh.coordinates() # from (implied) nm to m

    """
    Run the simulation.

    """
    print "Running the simulation..."

    llg = LLG(mesh)
    llg.alpha = 0.1
    llg.Ms = 0.86e6 # A/m
    llg.C = 1.3e-11 # J/m
    llg.H_app = (0.43e6, 0, 0) # A/m
    llg.set_m0(("2*x[0]/L - 1","2*x[1]/W - 1","1"), L=3e-8, H=1e-8, W=1e-8)
    llg.setup(exchange_flag=True)

    llg_wrap = lambda t, y: llg.solve_for(y, t)
    t0 = 0; dt = 1e-11; tmax = 1e-9 # s
    r = ode(llg_wrap).set_integrator("vode", method="bdf", rtol=1e-3)
    r.set_initial_value(llg.m, t0)

    fh = open(MODULE_DIR + "/averages.txt", "w")
    while r.successful() and r.t <= tmax:
        mx, my, mz = llg.m_average
        fh.write(str(r.t) + " " + str(mx) + " " + str(my) + " " + str(mz) + "\n")
        r.integrate(r.t + dt)
    fh.close()

if __name__ == "__main__":
    run_simulation()
