"""
Test the convergence of the fembem solvers with a uniformly magnetized unit
sphere. At the moment high run time is expected so this is not yet a pytest.
"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

import pytest
from dolfin import *
import numpy as np
from finmag.demag import solver_fk, solver_gcr
from finmag.demag.problems import prob_fembem_testcases as pft
import finmag.util.error_norms as en
import test_solvers as ts

#Initialize problems
#These problems should contain a sequence of finer meshes
#Refinement is to be avoided as it does not improve the polygonal boundary
#approximation to a sphere
finenesslist = range(2,10)
problems = [pft.MagUnitSphere(n) for n in finenesslist]

#FK Solver data
solvers_fk = [solver_fk.FemBemFKSolver(p) for p in problems]
phi_fk = [s.solve() for s in solvers_fk]
Hdemag_fk = [s.get_demagfield(s.phi) for s in solvers_fk]

#GCR Solver data
solvers_gcr = [solver_gcr.FemBemGCRSolver(p) for p in problems]
phi_gcr = [s.solve() for s in solvers_gcr]
Hdemag_gcr = [s.get_demagfield(s.phitot) for s in solvers_gcr]

class UnitSphere_Analytical(object):
    """
    Class containing information regarding the 3d analytical solution of a Demag Field in a uniformly
    demagnetized unit sphere with M = (1,0,0)
    """
    def __init__(self,V,VV):
        self.potential = project(Expression("-x[0]/3.0"),V)
        self.Hdemag = project(Expression(("1.0/3.0","0.0","0.0")),VV)

#Analytical Solution data assuming FK and GCR use the same function spaces
solvers_ana = [UnitSphere_Analytical(phi_gcr[i].function_space(),Hdemag_gcr[i].function_space()) for i in range(len(problems))]
phi_ana = [s.potential for s in solvers_ana]
Hdemag_ana = [s.Hdemag for s in solvers_ana]

def convergence_test(fn_approx, fn_ana, norm):
    """
    Test the convergence of a sequence of solutions
    to some sequence of analytical values in a norm

    fn_approx = list of dolfin functions
    fn_ana = list of dolfin functions
    norm = function that takes two dolfin functions as arguments
    """

    tups = list(zip(fn_approx, fn_ana))
    tups.reverse()
    errors = [norm(s,a) for s,a in tups]
    return errors

#Output Report
starline = "*****************************************************"
print starline
print "Report of convergence for the case of a magnetized unit sphere"
print starline
print "Mesh Fineness"
print finenesslist
print
print starline
print "FK Solver L2 potential"
print convergence_test(phi_fk,phi_ana, en.L2_error)
print
print "FK Solver L2 Hdemag"
print convergence_test(Hdemag_fk,Hdemag_ana, en.L2_error)
print
print "FK Solver max potential"
print convergence_test(phi_fk,phi_ana, en.discrete_max_error)
print 
print "FK Solver max demag x"
print convergence_test([p.split(True)[0] for p in Hdemag_fk],[p.split(True)[0] for p in Hdemag_ana], en.discrete_max_error)
print 
print "FK Solver max demag y"
print convergence_test([p.split(True)[1] for p in Hdemag_fk],[p.split(True)[1] for p in Hdemag_ana], en.discrete_max_error)
print
print "FK Solver max demag z"
print convergence_test([p.split(True)[2] for p in Hdemag_fk],[p.split(True)[2] for p in Hdemag_ana], en.discrete_max_error)
print 

print starline
print "GCR Solver L2 potential"
print convergence_test(phi_gcr,phi_ana, en.L2_error)
print
print "GCR Solver L2 Hdemag"
print convergence_test(Hdemag_gcr,Hdemag_ana, en.L2_error)
print
print "GCR Solver max potential"
print convergence_test(phi_gcr,phi_ana, en.discrete_max_error)
print 
print "GCR Solver max demag x"
print convergence_test([p.split(True)[0] for p in Hdemag_gcr],[p.split(True)[0] for p in Hdemag_ana], en.discrete_max_error)
print 
print "GCR Solver max demag y"
print convergence_test([p.split(True)[1] for p in Hdemag_gcr],[p.split(True)[1] for p in Hdemag_ana], en.discrete_max_error)
print
print "GCR Solver max demag z"
print convergence_test([p.split(True)[2] for p in Hdemag_gcr],[p.split(True)[2] for p in Hdemag_ana], en.discrete_max_error)
print 

