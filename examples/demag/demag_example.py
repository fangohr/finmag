import numpy as np
import dolfin as df
from finmag.demag.solver_fk import FemBemFKSolver
from finmag.util.convert_mesh import convert_mesh

mesh = df.UnitSphere(10)
#mesh = df.Mesh(convert_mesh("sphere1.geo"))
V = df.VectorFunctionSpace(mesh, "CG", 1)
m = df.interpolate(df.Constant((1,0,0)), V)
Ms = 1
#solver = FemBemFKSolver(V, m, Ms)

#from finmag.demag.problems.prob_fembem_testcases import FemBemDeMagProblem
#Why didn't that work??
class FemBemDeMagProblem:
    def __init__(self, mesh, m):
        self.mesh = mesh
        self.M = m
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
