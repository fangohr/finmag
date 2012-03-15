"""Main Module to run the other modules from"""
from solver_nitsche import *
from solver_gcr import *
from prob_trunc_testcases import *
from dolfin import *

truncproblem = MagUnitCircle()
#fembemproblem = truncproblem.create_fembem_problem()

#print truncproblem.coremesh.coordinates()
#print fembemproblem.mesh.coordinates()
nitschesolver = NitscheSolver(truncproblem)
#gcrsolver = GCRFemBemDeMagSolver(fembemproblem)
nitschesolver.solve()
plot(nitschesolver.phitot,title = "Nitsche Solution")
#plot (gcrsolver.solve(),title = "CGR Solution")
interactive()


