import numpy as np
import dolfin as df
from finmag.util.convert_mesh import convert_mesh
#from finmag.demag.solver_gcr import FemBemGCRSolver
#from finmag.demag.solver_fk_test import SimpleFKSolver
from finmag.demag.solver_fk import FemBemFKSolver
#from finmag.demag.problems import FemBemDeMagProblem
import pylab as p
import sys, os, commands, subprocess
from finmag.sim.llg import LLG

class FemBemDeMagProblem(object):
    """Have no idea why I can't import this now.."""
    def __init__(self, mesh, m):
        self.mesh = mesh
        self.M = m
        self.Ms = 1


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
nmagoutput = os.path.join(MODULE_DIR, 'nmag_data.dat')

# Need a clean file
if os.path.isfile(nmagoutput):
    os.remove(nmagoutput)

vertices = []
xavg = []
xmax = []
xmin = []
ymax = []
zmax = []
stddev = []
errorH = []
maxerror = []
nxmax = []
nxavg = []
nstddev = []
errnorm = []

#for maxh in (2, 1, 0.8, 0.7):
for maxh in (5, 3, 2, 1.5):

    # Create geofile
    geo = """
algebraic3d

solid main = sphere (0, 0, 0; 10)-maxh=%s ;

tlo main;""" % str(maxh)
    absname = "sphere_maxh_%s" % str(maxh)
    geofilename = os.path.join(MODULE_DIR, absname)
    geofile = geofilename + '.geo'
    f = open(geofile, "w")
    f.write(geo)
    f.close()

    # Finmag data
    mesh = df.Mesh(convert_mesh(geofile))
    #mesh.coordinates()[:] = mesh.coordinates()[:]*1e-9 #this makes the results worse!!! HF
    print "Using mesh with %g vertices" % mesh.num_vertices()
    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)

    # Old code
    """
    M = ("1.0", "0.0", "0.0")
    problem = FemBemDeMagProblem(mesh, M)
    solver = FemBemGCRSolver(problem)
    phi = solver.solve()
    H_demag = df.project(-df.grad(phi), V)
    """

    # Weiwei code
    """
    m = df.interpolate(df.Constant((1,0,0)), V)
    Ms = 1
    solver = SimpleFKSolver(V, m, Ms)
    H_demag = df.Function(V)
    demag = solver.compute_field()
    H_demag.vector()[:] = demag

    x, y, z = H_demag.split(True)
    x, y, z = x.vector().array(), y.vector().array(), z.vector().array()
    """

    m = df.interpolate(df.Constant((1,0,0)), V)
    problem = FemBemDeMagProblem(mesh, m)
    solver = FemBemFKSolver(problem)
    H_demag = df.Function(V)
    demag = solver.compute_field()
    H_demag.vector()[:] = demag
    demag.shape = (3, -1)
    x, y, z = demag[0], demag[1], demag[2]

    # Find #vertices, x max and x avg
    vertices.append(mesh.num_vertices())
    xavg.append(np.average(x))
    xmax.append(max(x))
    xmin.append(min(x))
    ymax.append(max(abs(y)))
    zmax.append(max(abs(z)))

    # Find standard deviation
    func = H_demag.vector().array()
    N = len(func)
    exct = np.zeros(N)
    exct[:len(x)] = -1./3*np.ones(len(x))
    sdev = np.sqrt(1./N*sum((func - exct)**2))
    stddev.append(sdev)

    # Find errornorm
    exact = df.interpolate(df.Constant((-1./3, 0, 0)), V)
    sphere_volume=4/3.*np.pi*(10)**3
    errnorm.append(df.errornorm(H_demag, exact, mesh=mesh)/sphere_volume)

    #actual error:
    tmperror = func-exct
    tmpmaxerror = max(abs(tmperror))
    errorH.append(tmperror)
    maxerror.append(tmpmaxerror)

    # Nmag data
    if subprocess.call(["which", "nsim"]) == 0:
        print "Running nmag now."
        has_nmag = True
    else:
        has_nmag = False
        continue

    # Create neutral mesh
    cmd1 = 'netgen -geofile=%s -meshfiletype="Neutral Format" -meshfile=%s.neutral -batchmode' % (geofile, geofilename)
    status, output = commands.getstatusoutput(cmd1)
    #if status != 0:
    #    print 'Netgen failed. Aborted.'
    #    sys.exit(1)

    # Convert neutral mesh to nmag type mesh
    cmd2 = 'nmeshimport --netgen %s.neutral %s.nmesh.h5' % (geofilename, geofilename)
    status, output = commands.getstatusoutput(cmd2)
    if status != 0:
        print 'Nmeshimport failed. Aborted.'
        sys.exit(2)

    # Run nmag
    cmd3 = 'nsim run_nmag.py --clean %s.nmesh.h5 nmag_data.dat' % geofilename
    status, output = commands.getstatusoutput(cmd3)
    if status != 0:
        print output
        print 'Running nsim failed. Aborted.'
        sys.exit(3)
    print "\nDone with nmag."

# Extract nmag data
if has_nmag:
    f = open('nmag_data.dat', 'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.split()
        if len(line) == 3:
            nxavg.append(float(line[0]))
            nxmax.append(float(line[1]))
            nstddev.append(float(line[2]))

# Plot
p.plot(vertices, xavg, 'x--',label='Finmag x-avg')
p.plot(vertices, xmax, 'o-',label='Finmag x-max')
p.plot(vertices, xmin, '^:',label='Finmag x-min')

if has_nmag:
    p.plot(vertices, nxavg, label='Nmag x-avg')
    p.plot(vertices, nxmax, label='Nmag x-max')
    p.title('Nmag - Finmag comparisson')
else:
    p.title('Finmag x vs vertices')

p.xlabel('vertices')
p.grid()
p.legend()
p.savefig(os.path.join(MODULE_DIR, 'xvalues.png'))

p.figure()
p.plot(vertices, stddev, label='Finmag standard deviation')

if has_nmag:
    p.plot(vertices, nstddev, label='Nmag standard deviation')

p.xlabel('vertices')
p.title('Standard deviation')
p.grid()
p.legend()
p.savefig(os.path.join(MODULE_DIR, 'stddev.png'))

p.figure()
p.plot(vertices, errnorm, label='Finmag errornorm')
p.xlabel('vertices')
p.title('Error norm')
p.grid()
p.legend()
p.savefig(os.path.join(MODULE_DIR, 'errnorm.png'))


p.figure()
p.plot(vertices, maxerror, 'o-',label='Finmag maxerror H_demag-x')
p.plot(vertices, ymax, 'x-',label='Finmag maxerror H_demag-y')
p.plot(vertices, zmax, '^-',label='Finmag maxerror H_demag-z')
p.xlabel('vertices')
p.title('Max Error per component')
p.grid()
p.legend()
p.savefig(os.path.join(MODULE_DIR, 'maxerror.png'))


p.figure()
p.loglog(vertices, stddev, label='Finmag standard deviation')

if has_nmag:
    p.loglog(vertices, nstddev, label='Nmag standard deviation')

p.xlabel('vertices')
p.title('Standard deviation (log-log)')
p.grid()
p.legend()
p.savefig(os.path.join(MODULE_DIR, 'stddev_loglog.png'))

print "Useful plots: maxerror.png, stddev.png, xvalues.png"
