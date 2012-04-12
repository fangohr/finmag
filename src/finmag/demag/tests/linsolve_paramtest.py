"""
This script contains timings for the FEM parts of the FEMBEM demag solvers.
Different linear algebra solution methods are tested along with preconditioners.

The class LinAlgDemagTester is the interface for this module.
""" 

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

import dolfin as df
import finmag.demag.problems.prob_base as pb
from finmag.util.timings import Timings
from finmag.demag.solver_gcr import FemBemGCRSolver
from finmag.demag.solver_fk import FemBemFKSolver

class FemBemGCRSolverLinalgTime(FemBemGCRSolver):
    """GCR solver with timings for linear solve"""
    
    def __init__(self,problem):
        FemBemGCRSolver.__init__(self,problem)
        #Switch off the BEM countdown
        self.countdown = False

    def setparam(self,p1,p2,timer):
        self.phiasolverparams = p1
        self.phibsolverparams = p2
        self.timer = timer

    def solve_phia(self):
        self.timer.start("linsolve solve_phia")
        r =FemBemGCRSolver.solve_phia(self)
        self.timer.stop("linsolve solve_phia")
        return r
    
    def solve_laplace_inside(self,function,solverparams):
        self.timer.start("linsolve solve_phib")
        r =FemBemGCRSolver.solve_laplace_inside(self,function,solverparams)
        self.timer.stop("linsolve solve_phib")                                                  
        return r

    def report(self,n = 10):
        print "\n Linear solve timings of the GCR demag solver"
        print "mesh size in verticies = ",self.problem.mesh.num_vertices()
        print "phia parameters"
        print self.phiasolverparams
        print "phib parameters"
        print self.phibsolverparams
        print "\n",self.timer.report_str(n)


class FemBemFKSolverLinalgTime(FemBemFKSolver):
    """FK solver with timings for linear solve"""
    
    def __init__(self,problem,timer):
        FemBemFKSolver.__init__(self,problem)
        #Switch off the BEM countdown
        self.countdown = False

    def setparam(self,p1,p2):
        self.phi1solverparams = p1
        self.phi2solverparams = p2
        self.timer = timer

    def compute_phi1(self, M, V,solverparams):
        self.timer.start("linsolve solve_phi1")
        FemBemFKSolver.solve_laplace_inside(self,function,solverparams)
        self.timer.stop("linsolve solve_phi1")                                                  
    
    def solve_laplace_inside(self,function,solverparams):
        self.timer.start("linsolve solve_phi2")
        FemBemFKSolver.solve_laplace_inside(self,function,solverparams)
        self.timer.stop("linsolve solve_phi2")                                                  

    def report(self,n = 10):
        print "Linear solve timings of the FK demag solver"
        print "mesh size in verticies = ",self.problem.mesh.num_vertices()
        print "phi1 parameters"
        print self.phi1solverparams
        print "phi2 parameters"
        print self.phi2solverparams
        print "\n",self.timer.report_str(n)


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
                     
        Note: Running groups of testparams instead of indidual params saves on BEM assembly

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

##Default linear solver parameters to test.
default_params =[{"linear_solver":"lu"},    \
                 {"linear_solver":"gmres","preconditioner": "ilu"}, \
                 {"linear_solver":"cg","preconditioner": "ilu"}]
                 #{"linear_solver":"gmres","preconditioner":"ilu","absolute_tolerance":1.0e-5} ]


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
    sizelist = [5.0,4.0,3.0,2.0,1.0]
    problems = [pft.MagSphere(10,hmax = i) for i in sizelist]

    #Run the tests
    testers = [LinAlgDemagTester(p.mesh,p.M,"GCR",default_params) for p in problems]
    for t in testers:
        t.test()

    #Now create a plot with meshsize vs.linear solve time.
    meshsizes = [p.mesh.num_vertices() for p in problems]
    lutime = [t.timelist[0].recorded_sum() for t in testers]
    gmirutime = [t.timelist[1].recorded_sum() for t in testers]
    cgruntime = [t.timelist[2].recorded_sum() for t in testers]

    plt.plot(meshsizes,lutime,label = "lu")
    plt.plot(meshsizes,gmirutime,label = "gmres-ilu")
    plt.plot(meshsizes,cgruntime,label = "cg-ilu")
    plt.xlabel("Number of Mesh vertices")
    plt.ylabel("solver time (s)")
    plt.title("Demag Linear Solver times")
    plt.legend()
    plt.show()
    
##    
##        #Generate the plots
##        figurenum = 2
##        #For every error-norm combination we want to plot...
##        for i,errorname in enumerate(self.errordata[self.errordata.keys()[0]]):
##            if i <> 0 and i % 4 ==0:
##                #The subplots are full so start a new page
##                plt.figure(figurenum)
##                figurenum += 1
##            #put the subplot in position i modulo 4 + 1
##            plotnum = self.subplots + str(i%4+1)
##            plt.subplot(*plotnum)
##            
##            #... the solutions errors for evey solver
##            for j,sol in enumerate(self.test_solver_classes):
##                #TODO plot the error data correctly
##                plt.plot(self.xaxis[1],self.errordata[sol][errorname],self._styledic[j],label = sol)
##            #Give the plot titles
##            plt.title(errorname)
##            plt.xlabel(self.xaxis[0])
##            plt.ylabel("Error")
##            plt.legend()
##        plt.show()
