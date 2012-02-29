"""Main Module to run the other modules from"""
from solver_nitsche import *
from solver_fk import *
from prob_testcases import *
from dolfin import *

problem = MagUnitCircle()
solver = NitscheSolver(problem)
solution = solver.solve()
plot(solver.phi_core)
Hdemag = grad(solver.phi_core)
Vec = VectorFunctionSpace(problem.coremesh,"CG",1)
Hdemag = project(Hdemag,Vec)
plot(Hdemag)
interactive()
