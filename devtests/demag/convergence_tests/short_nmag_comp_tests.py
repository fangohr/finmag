import numpy as np
import dolfin as df
from finmag.tests.demag.prob_base import DemagProblem
from finmag.util.convert_mesh import convert_mesh
from finmag.demag.solver_gcr import FemBemGCRSolver
import pylab as p
import sys, os, commands

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
nmagoutput = os.path.join(MODULE_DIR, 'nmag_data.dat')

# Need a clean file
if os.path.isfile(nmagoutput):
    os.remove(nmagoutput)

vertices = []
xavg = []
xmax = []
stddev = []
nxmax = []
nxavg = []
nstddev = []
errnorm = []

for maxh in (2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.2):
    
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
    print "Using mesh with %g vertices" % mesh.num_vertices()
    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    M = ("1.0", "0.0", "0.0")
    problem = DemagProblem(mesh, M)
    solver = FemBemGCRSolver(problem)
    phi = solver.solve()
    H_demag = df.project(-df.grad(phi), V)

    x, y, z = H_demag.split(True)
    x, y, z = x.vector().array(), y.vector().array(), z.vector().array()
    
    # Find #vertices, x max and x avg
    vertices.append(mesh.num_vertices())
    xavg.append(np.average(x))
    xmax.append(max(x))

    # Find standard deviation
    func = H_demag.vector().array()
    N = len(func)
    exct = np.zeros(N)
    exct[:len(x)] = -1./3*np.ones(len(x))
    sdev = np.sqrt(1./N*sum((func - exct)**2))
    stddev.append(sdev)

    # Find errornorm
    exact = df.interpolate(df.Constant((-1./3, 0, 0)), V)
    errnorm.append(df.errornorm(H_demag, exact, mesh=mesh))

    # Nmag data

    # Risky test...
    if 'nmag' in str(sys.path):
        has_nmag = True
        continue
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
	print 'Running nsim failed. Aborted.'
        sys.exit(3)
    
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
p.plot(vertices, xavg, label='Finmag x-avg')
p.plot(vertices, xmax, label='Finmag x-max')

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


