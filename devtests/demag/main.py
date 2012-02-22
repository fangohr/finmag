#Main Module to run the other modules from
from solver_nitsche import *
from solver_fk import *
from prob_testcases import *

problem = MagUnitSphere()
solver = NitscheSolver(problem)
solution = solver.solve()

soltrue = Expression("-x[0]/3.0")
soltrue = project(soltrue,solver.V)
l1form = abs(solution - soltrue)*problem.dxC
print assemble(l1form, cell_domains = problem.corefunc)
