"""
This script contains timings for the FEM parts of the FEMBEM demag solvers.
Different linear algebra solution methods are tested along with preconditioners.

The class LinAlgDemagTester is the interface for this module.
""" 

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
import finmag.demag.problems.prob_base as pb
from finmag.util.timings import Timings
from finmag.demag.solver_gcr import FemBemGCRSolver
from finmag.demag.solver_fk import FemBemFKSolver
from finmag.demag.solver_base import FemBemDeMagSolver

##Default linear solver parameters to test.
default_params =[{"linear_solver":"gmres","preconditioner": "ilu"}, \
                 {"linear_solver":"lu"}]    \
           #      {"linear_solver":"cg","preconditioner": "ilu"}]
                 #{"linear_solver":"gmres","preconditioner":"ilu","absolute_tolerance":1.0e-5} ]

class LinAlgDemagTester(object):
    """
    A class to test the speed of linalg solvers and preconditioners
    used in demag calculation. 
    """
    def __init__(self,mesh,M,solver,testparams):
        """
        M          - Magnetisation field
        solver     - "GCR" or "FK"
        testparams - A list of dictionaries containing parameters for the linear solve.
                     Can also give a tuple of two dictionaries if different parameters are
                     wished for the two linear solves.
                     
        Note: Running groups of testparams instead of individual params saves on BEM assembly.

        """
        self.testparams = testparams
        self.problem = pb.FemBemDeMagProblem(mesh,M)
        if solver == "FK":
            self.solver = FemBemFKSolverLinalgTime(self.problem)
        elif solver == "GCR":
            self.solver = FemBemGCRSolverLinalgTime(self.problem)
        else:
            raise Exception("Only 'FK, and 'GCR' solver values possible")
                    
    def test(self):

        #Create a list of timer objects, there will be 1 timer for each set of parameters
        self.timelist = []

        for test in self.testparams:
            #Two or one sets of parameters can be given. Depending on if we want to use
            #different solvers for different steps.
            try:
                t1,t2 = test[0],test[1]
            except KeyError:
                t1,t2 = test,test
            timer = Timings()
            self.solver.setparam(t1,t2,timer)
            self.solver.solve()
            self.solver.report()
            self.timelist.append(timer)
        #After the testing is finished delete the BEM to free up memory.
        del self.solver.bem

class LinAlgTimer(FemBemDeMagSolver):
    """Class containing shared methods for the GCR and FK linalg timing classes"""

    def linsolve_laplace_inside(self,function,laplace_A,solverparams = None):
        """
        Linear solve for laplace_inside written for the
        convenience of changing solver parameters in subclasses
        """
        self.timer.start("2nd linear solve")    
        function = FemBemDeMagSolver.linsolve_laplace_inside(self,function,laplace_A,solverparams)
        self.timer.stop("2nd linear solve")
        return function

    def setparam(self,p1,p2,timer):
        self.phi1solverparams = p1
        self.phi2solverparams = p2
        self.timer = timer
    
    def report(self,n = 10):
        print "".join(["Linear solve timings of the ",self.name])
        print "mesh size in verticies = ",self.problem.mesh.num_vertices()
        print "First solve parameters"
        print self.phi1solverparams
        print "Second solve parameters"
        print self.phi2solverparams
        print "\n",self.timer.report_str(n)

class FemBemGCRSolverLinalgTime(FemBemGCRSolver,LinAlgTimer):
    """GCR solver with timings for linear solve"""
    
    def __init__(self,problem):
        FemBemGCRSolver.__init__(self,problem)
        #Switch off the BEM countdown
        self.countdown = False
        self.name = "GCR Solver"

    def linsolve_phia(self,A,F):
        """Linear solve for phia"""
        self.timer.start("1st linear solve")
        FemBemGCRSolver.linsolve_phia(self,A,F)
        self.timer.stop("1st linear solve")
    
class FemBemFKSolverLinalgTime(FemBemFKSolver,LinAlgTimer):
    """FK solver with timings for linear solve"""
    
    def __init__(self,problem,timer):
        FemBemFKSolver.__init__(self,problem)
        #Switch off the BEM countdown
        self.countdown = False
        self.name = "FK Solver"
        
    def linsolve_phi1(self,a,f):
        # Solve for the DOFs in phi1
        """Linear solve for phia"""
        self.timer.start("1st linear solve")
        FemBemGCRSolver.linsolve_phi1(self,a,f)
        self.timer.stop("1st linear solve")


###################################################################
#To the user:
###################################################################

#This command prints out possible linalg solver parameters
##df.info(df.LinearVariationalSolver.default_parameters(), 1)

if __name__ == "__main__":
    #As a default plot a sequence of solver values for GCR with different meshes
    
    import finmag.demag.problems.prob_fembem_testcases as pft
    import matplotlib.pyplot as plt

    #Create a range of mesh sizes
    sizelist = [1.5,1.2]
    problems = [pft.MagSphere(10,hmax = i) for i in sizelist]

    #Run the tests
    testers = [LinAlgDemagTester(p.mesh,p.M,"GCR",default_params) for p in problems]
    for t in testers:
        t.test()

    #Now create a plot with meshsize vs.linear solve time.
    meshsizes = [p.mesh.num_vertices() for p in problems]
    lutime = [t.timelist[0].recorded_sum() for t in testers]
    gmirutime = [t.timelist[1].recorded_sum() for t in testers]
    #cgruntime = [t.timelist[2].recorded_sum() for t in testers]

    plt.plot(meshsizes,lutime,label = "lu")
    plt.plot(meshsizes,gmirutime,label = "gmres-ilu")
    #plt.plot(meshsizes,cgruntime,label = "cg-ilu")
    plt.xlabel("Number of Mesh vertices")
    plt.ylabel("solver time (s)")
    plt.title("Demag Linear Solver times")
    plt.legend()
    plt.show()
