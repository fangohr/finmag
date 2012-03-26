import numpy as np
from dolfin import Mesh, plot
from finmag.util.convert_mesh import convert_mesh
from finmag.demag.solver_gcr import FemBemGCRSolver
from finmag.demag.problems.prob_base import FemBemDeMagProblem

mesh = Mesh(convert_mesh("../../mesh/sphere10.geo"))
M = ("1.0", "0.0", "0.0")
problem = FemBemDeMagProblem(mesh, M)
solver = FemBemGCRSolver(problem)
phi = solver.solve()
H_demag = solver.get_demagfield(phi)


if __name__ == '__main__':
    p = plot(phi, interactive=True)
    #p.write_png("examples/demag/phi.png")

    solver.save_function(phi, "GCR_phi")
    solver.save_function(H_demag, "GCR_demagfield")

    x, y, z = H_demag.split(True)
    x, y, z = x.vector().array(), y.vector().array(), z.vector().array()
    for i in range(len(x)):
        print x[i], y[i], z[i]

    print "Max values: x:%g, y:%g, z:%g" % (max(x), max(y), max(z))
    print "Min values: x:%g, y:%g, z:%g" % (min(x), min(y), min(z))
    print "Avg values: x:%g, y:%g, z:%g" % (np.average(x), np.average(y), np.average(z))
