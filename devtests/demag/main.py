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
Hdemagproj = solver.Hdemag_core

print "Projection Gradient of potential function at 0.1,0,0",Hdemagproj((0.0,0,0))
solver.save_function(Hdemagproj,"Hdemag")
##demagfile = File("results/demag.pvd")
##demagfile << Hdemag
##interactive()
