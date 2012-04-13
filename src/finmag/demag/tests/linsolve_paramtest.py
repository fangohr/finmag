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

##Default linear solver parameters to test.
default_params =[{"linear_solver":"gmres","preconditioner": "ilu"}, \
                 {"linear_solver":"lu"},    \
                 {"linear_solver":"cg","preconditioner": "ilu"}]
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

class LinAlgTimer(object):
    """Class containing shared methods for the GCR and FK linalg timing classes"""
    
    def solve_laplace_inside(self,function,solverparams):                                                          
        """Take a functions boundary data as a dirichlet BC and solve
            a laplace equation"""
        
        bc = DirichletBC(self.V,function, "on_boundary")
        #Buffer data independant of M
        if not hasattr(self,"poisson_matrix"):
            self.build_poisson_matrix()
        if not hasattr(self,"laplace_F"):
            #RHS = 0
            self.laplace_f = Function(self.V).vector()

        #Copy the poisson matrix it is shared and should
        #not have bc applied to it.
        laplace_A = self.poisson_matrix
        #Apply BC
        bc.apply(laplace_A)
        #Boundary values of laplace_f are overwritten on each call.
        bc.apply(self.laplace_f)
        self.timer.start("linsolve 2nd solve")
        solve(laplace_A,function.vector(),\
              self.laplace_f,\
              solver_parameters = self.phi2solverparams)
        self.timer.stop("linsolve 2nd solve")
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

    def solve_phia(self):
        V = self.phia.function_space()
      
        #Buffer data independant of M
        if not hasattr(self,"formA_phia"):
           #Try to use poisson_matrix if available
           if not hasattr(self,"poisson_matrix"):
                self.build_poisson_matrix()
           self.phia_formA = self.poisson_matrix
           #Define and apply Boundary Conditions
           self.phia_bc = DirichletBC(V,0,"on_boundary")
           self.phia_bc.apply(self.phia_formA)               
           
        #Source term depends on M
        f = (-div(self.M)*self.v)*dx  #Source term
        F = assemble(f)
        self.phia_bc.apply(F)

        #Solve for phia
        A = self.phia_formA
        self.timer.start("linsolve 1st solve")
        solve(A,self.phia.vector(),F, \
        solver_parameters = self.phi1solverparams)
        self.timer.stop("linsolve 1st solve")


class FemBemFKSolverLinalgTime(FemBemFKSolver,LinAlgTimer):
    """FK solver with timings for linear solve"""
    
    def __init__(self,problem,timer):
        FemBemFKSolver.__init__(self,problem)
        #Switch off the BEM countdown
        self.countdown = False
        self.name = "FK Solver"

    def compute_phi1(self, M, V,solverparams):
        # Define functions
        n = FacetNormal(V.mesh())
        
        # Define forms
        eps = 1e-8
        a = dot(grad(self.u),grad(self.v))*dx - dot(eps*self.u,self.v)*dx 
        f = dot(n, self.M)*self.v*ds - div(self.M)*self.v*dx

        # Solve for the DOFs in phi1
        self.timer.start("linsolve 1st solve")
        solve(a == f, self.phi1,solver_parameters = self.phi1solverparams)
        self.timer.stop("linsolve 2nd solve")                                                  


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
    sizelist = [4.0,3.0,2.0]
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
