"""
This script contains timings for the FEM parts of the FEMBEM demag solvers.
Different linear algebra solution methods are tested along with preconditioners.

The class LinAlgDemagTester is the interface for this module.
""" 

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

import numpy
from dolfin import *
import finmag.demag.problems.prob_base as pb
from finmag.util.timings import Timings
from finmag.demag.solver_gcr import FemBemGCRSolver
from finmag.demag.solver_fk import FemBemFKSolver
from finmag.demag.solver_base import FemBemDeMagSolver


#Generate all possible solver parameters
def solver_parameters(solver_exclude, preconditioner_exclude):
    linear_solver_set = ["lu"] 
    linear_solver_set += [e[0] for e in dolfin.krylov_solver_methods()]
    preconditioner_set = [e[0] for e in dolfin.krylov_solver_preconditioners()]

    solver_parameters_set = []
    for l in linear_solver_set:
        if l in solver_exclude:
            continue
        for p in preconditioner_set:
            if p in preconditioner_exclude:
                continue
            if (l == "lu" or l == "default") and p != "none":
                continue
            solver_parameters_set.append({"linear_solver": l, "preconditioner": p})
    return solver_parameters_set

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
    #Solver type for use in the script, "FK" or "GCR".
    fembemsolvertype = "GCR"

    ##Default linear solver parameters to test.
    ## Each entry can either be a single dictionary or a tuple/list of dictionaries.
    ## In the single case both linear solves will be computed using the specified parameters.
    ## If a tuple/list is given the 1st solve is done according to the 1st parameter dictionary,
    ## and the 2nd solve according to the 2ns parameter dictionary.

    default_params =[{"linear_solver":"gmres","preconditioner": "ilu"}, \
                     {"linear_solver":"lu"},    \
                     {"linear_solver":"cg","preconditioner": "ilu"},
                     {"linear_solver":"gmres","preconditioner":"ilu","absolute_tolerance":1.0e-5},
                     {"linear_solver":"gmres","preconditioner":"ilu","relative_tolerance":1.0e-5}]

    #to use all methods, activate next line
    #default_params = solver_parameters(solver_exclude=[], preconditioner_exclude=[])
    #print default_params


    #As a default plot a sequence of solver values for GCR with different meshes
    
    import finmag.demag.problems.prob_fembem_testcases as pft
    import matplotlib.pyplot as plt

    #Create a range of mesh sizes
    sizelist = [4,2,1.5]
    #Important use floats here so the entry order stays consistent 
    sizelist = ([4.0,3.0,2.0,1.5,1.0])
    problems = [pft.MagSphere(10,hmax = i) for i in sizelist]

    #Run the tests
    testers = [LinAlgDemagTester(p.mesh,p.M,fembemsolvertype,default_params) for p in problems]
    for t in testers:
        t.test()
        
    meshsizes = numpy.array([p.mesh.num_vertices() for p in problems])
#For each linear solve..
    solvelist = ["1st linear solve","2nd linear solve"]
    for j,linearsolve in enumerate(solvelist):

        #Now create a plot with meshsize vs.linear solve time.

        print ("%d testers, %d parameter sets" % (len(testers),len(default_params)))
        for i,para in enumerate(default_params):
            try:
                #There are two seperate linear solve entries
                params = para[j]
            except:
                #Just one entry
                params = para
                
            itime = numpy.array([t.timelist[i].gettime(linearsolve) for t in testers])/meshsizes
            if not 'preconditioner' in params.keys():
                precon='(none)'
            else:
                precon = params['preconditioner']
            solver=params['linear_solver']
            other = "".join( "%8s:%s " % (key,value) for key,value in params.items() if key not in ['preconditioner','linear_solver'])
            label='solver: %10s, precond:  %10s %s' % (solver,precon,other)
            print "Plotting %s" % label
            plt.plot(meshsizes,itime,label = label)

        plt.xlabel("Number of Mesh vertices")
        plt.ylabel("solver time (s) per mesh node")
        plt.title(" ".join(["Demag",fembemsolvertype,linearsolve,"times"]))
        plt.legend(loc=0)
        #Create a new figure for each linear solve
        if j != len(solvelist) - 1:
            plt.figure()
    plt.show()
