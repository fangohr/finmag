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
problems = [pft.MagUnitSphere(n) for n in range(2,8)]

#Analytical Solution data
solvers_ana = [ts.UnitSphere_Analytical(p.mesh) for p in problems]
phi_ana = [s.potential for s in solvers_ana]
Hdemag_ana = [s.Hdemag for s in solvers_ana]
print phi_ana

#FK Solver data
solvers_fk = [solver_fk.FemBemFKSolver(p) for p in problems]
phi_fk = [s.solve() for s in solvers_fk]
Hdemag_fk = [s.get_demagfield(s.phi) for s in solvers_fk]

#GCR Solver data
solvers_gcr = [solver_gcr.GCRFemBemDeMagSolver(p) for p in problems]
phi_gcr = [s.solve() for s in solvers_gcr]
Hdemag_gcr = [s.get_demagfield(s.phitot) for s in solvers_gcr]

def convergence_test(fn_approx, fn_ana, norm):
    """
    Test the convergence of a sequence of solutions
    to some sequence of analytical values in a norm

    fn_approx = list of dolfin functions
    fn_ana = list of dolfin functions
    norm = function that takes two dolfin functions as arguments
    """
    errors = [norm(s,a) for s,a in zip(fn_approx, fn_ana)]
    return errors

#Output Report
starline = "*****************************************************"
print starline
print "Report of convergence for the case of a magnetized unit sphere"
print starline
print
print 
print "FK Solver L2 potential"
print convergence_test(phi_fk,phi_ana, en.L2_error)
print
print "FK Solver L2 Hdemag"
print convergence_test(Hdemag_fk,Hdemag_ana, en.L2_error)
print
print "GCR Solver L2 potential"
print convergence_test(phi_gcr,phi_ana, en.L2_error)
print
print "GCR Solver L2 Hdemag"
print convergence_test(Hdemag_gcr,Hdemag_ana, en.L2_error)
