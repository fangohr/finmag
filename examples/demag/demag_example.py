import numpy as np
from dolfin import *
from finmag.util.convert_mesh import convert_mesh
from finmag.demag.solver_fk import FemBemFKSolver
from finmag.demag.problems.prob_base import FemBemDeMagProblem

mesh = UnitSphere(10)
m = interpolate(Constant((1,0,0)), VectorFunctionSpace(mesh, "CG", 1))
problem = FemBemDeMagProblem(mesh, m)
problem.Ms = 1
solver = FemBemFKSolver(problem)
H_demag = solver.compute_field()
H_demag.shape = (3, -1)

if __name__ == '__main__':
    x, y, z = H_demag
    print "Max values: x:%g, y:%g, z:%g" % (max(x), max(y), max(z))
    print "Min values: x:%g, y:%g, z:%g" % (min(x), min(y), min(z))
    print "Avg values: x:%g, y:%g, z:%g" % (np.average(x), np.average(y), np.average(z))
