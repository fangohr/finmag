"""Main Module to run the other modules from"""
from solver_nitsche import *
from solver_fk import *
from prob_testcases import *
from dolfin import *

problem = MagUnitSphere()
solver = NitscheSolver(problem)
solution = solver.solve()

print "s at (0, 0, 0):", solution((0.0, 0.0, 0.0))

print "s at (0, 0, 0.1):", solution((0.0, 0.0, 0.1))


##plot(solver.phi_core, title = "potential function")
demag_space = VectorFunctionSpace(problem.coremesh,"DG",0)
Hdemag = -grad(solver.phi_core)
Hdemagproj = project(Hdemag,demag_space)

print "Projection Gradient of potential function at 0.1,0,0",Hdemagproj((0.0,0,0))
##demagfile = File("results/demag.pvd")
##demagfile << Hdemag
##interactive()
