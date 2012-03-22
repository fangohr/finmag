"""
Test the convergence of the fembem solvers with a uniformly magnetized unit
sphere. At the moment high run time is expected so this is not yet a pytest. 

If you want to test a new solver (of class FemBemDeMagSolver) just add
the solver class to the dictionary "solvers"
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
import finmag.demag.solver_base as sb
import finmag.util.error_norms as en
import test_solvers as ts
import matplotlib.pyplot as plt

#This class mimics a FemBemDeMagSolver but returns analytical values.
class FemBemAnalytical(sb.FemBemDeMagSolver):
    """
    Class containing information regarding the 3d analytical solution of a Demag Field in a uniformly
    demagnetized unit sphere with M = (1,0,0)
    """
    def __init__(self,problem):
        super(FemBemAnalytical,self).__init__(problem)
        
    def solve(self):
        self.phi = project(Expression("-x[0]/3.0"),self.V)
        return self.phi
    
    def get_demagfield(self,phi):
        self.Hdemag = project(Expression(("1.0/3.0","0.0","0.0")),self.Hdemagspace)
        return self.Hdemag

############################################
#Section 0 Problems and solvers
############################################

###########################
#This controls the fineness of the meshes
# At the momenet UnitSphere(i)
#where i is in fineensslist
###########################   
finenesslist = range(2,4)
###########################

problems = [pft.MagUnitSphere(n) for n in finenesslist]
numelement = [p.mesh.num_cells() for p in problems]

#Master list of solvers####
solvers = {"FK Solver": solver_fk.FemBemFKSolver,"GCR Solver": solver_gcr.GCRFemBemDeMagSolver,"Analytical":FemBemAnalytical}
###########################

a = FemBemAnalytical(problems[0])

############################################
#Section 1 Functions
############################################

def componentlists(flist):
    """
    Take a list of functions and return
    3 lists of 3 subfunctions x,y,z
    """
    return [[f.split(True)[i] for f in flist] for i in range(3)]

def build_solver_data(solverclass,problems):
    """
    Takes a Solver class and a list of problems
    and returns a dictionary of solutions.
    """
    solvers = [solverclass(p) for p in problems]
    #The various solutions
    phi = [s.solve() for s in solvers]
    Hdemag = [s.get_demagfield(s.phi) for s in solvers]
    #Split the demag field into components
    Hdemagsub = componentlists(Hdemag)
    return {"solobj":solvers,"Potential":phi,"Hdemag":Hdemag,"Hdemagx":Hdemagsub[0], \
            "Hdemagy":Hdemagsub[1],"Hdemagz":Hdemagsub[2]}

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

def error_norms(sd1,sd2):
    """
    Calculate the various errors that we are interested in,
    comparing the functions in sd1 to those of sd2.
    sd = solver_data.
    """
    return {"Potential L2": convergence_test(sd1["Potential"],sd2["Potential"], en.L2_error), \
           "Hdemag L2": convergence_test(sd1["Hdemag"],sd2["Hdemag"], en.L2_error), \
           "Potential Discrete Max": convergence_test(sd1["Potential"],sd2["Potential"], en.discrete_max_error), \
           "Hdemag x Discrete Max": convergence_test(sd1["Hdemagx"],sd2["Hdemagx"], en.discrete_max_error), \
           "Hdemag y Discrete Max": convergence_test(sd1["Hdemagy"],sd2["Hdemagy"], en.discrete_max_error), \
           "Hdemag z Discrete Max": convergence_test(sd1["Hdemagz"],sd2["Hdemagz"], en.discrete_max_error)}

############################################
#Section 2 Data
############################################

#Initialize problems
#These problems should contain a sequence of finer meshes
#Refinement is to be avoided as it does not improve the polygonal boundary
#approximation to a sphere

#Get the functions we are interested in
solverdata = {k:build_solver_data(solvers[k],problems) for k in solvers}
#Compute error norms of functions compared to the analytical solution.
errornorms = {k:error_norms(solverdata[k], solverdata["Analytical"]) for k in solverdata if k <> "Analytical"}

#Output Report
starline = "*****************************************************"
print starline
print "Report of convergence for the case of a magnetized unit sphere"
print starline
print "Mesh Fineness"
print finenesslist

for solvername in errornorms:
    #name of solver
    print starline
    print "Convergence of ", solvername
    print starline
    #Names of norms and their values
    for errorname in errornorms[solvername]:
        print errorname
        print errornorms[solvername][errorname]
        print

#Outpot Plots
gcrdata = errornorms["GCR Solver"]["Potential L2"]
fkdata = errornorms["FK Solver"]["Potential L2"]

#This specificies the number of rows and columns of subplots
plotgrid = "23"
for i,errorname in enumerate(errornorms["GCR Solver"]):
    plotnum = plotgrid + str(i+1)

    plt.subplot(*plotnum)
    plt.plot(numelement,errornorms["GCR Solver"][errorname],"ro",label = "GCR")
    plt.plot(numelement,errornorms["FK Solver"][errorname],"bs",label = "FK")
    plt.title(errorname)
    plt.xlabel("Number of elements")
    plt.ylabel("Error")
    plt.legend()
plt.show()

