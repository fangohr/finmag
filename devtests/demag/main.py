#Main Module to run the other modules from
from solver_nitsche import *
from prob_testcases import *

problem = MagUnitCircle()
solver = NitscheSolver(problem, degree = 1)
solution = solver.solve()
gradphi = grad(solution)
demag = project(gradphi, solver.Mspace)
plot(solution, title = "phi")
plot(demag, title = "demag field")
interactive()
