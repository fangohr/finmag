"""Main Module to run the other modules from"""
from solver_nitsche import *
from solver_gcr import *
from prob_trunc_testcases import *
import prob_fembem_testcases as pft
from dolfin import *

#truncproblem = MagUnitCircle()
#fembemproblem = truncproblem.create_fembem_problem()
fembemproblem = pft.MagUnitCircle(30)
##print truncproblem.coremesh.coordinates()
##print fembemproblem.mesh.coordinates()
#nitschesolver = NitscheSolver(truncproblem)
gcrsolver = GCRFemBemDeMagSolver(fembemproblem)
gcrsol = gcrsolver.solve()
#nitschesolver.solve()
#plot(nitschesolver.phitot,title = "Nitsche Solution")
plot (gcrsol ,title = "CGR Solution")
interactive()


