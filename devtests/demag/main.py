#Main Module to run the other modules from
from solver_nitsche import *
from prob_testcases import *

problem = MagUnitSphere()
solver = NitscheSolver(problem)
solution = solver.solve()
plot(solution)
interactive()
