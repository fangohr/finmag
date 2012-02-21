#Main Module to run the other modules from
from solver_nitsche import *
from solver_fk import *
from prob_testcases import *

problem = MagUnitInterval()
solver = FKSolver(problem)
solution = solver.solve()

##solver = NitscheSolver(problem, degree = 1)
##solution = solver.solve()
##gradphi = grad(solution)
##demag = project(gradphi, solver.Mspace)
plot(solution, title = "phitotal")
plot(solver.phi0, title = "phi0")
plot(solvsolverve
####plot(demag, title = "demag field")
interactive()
