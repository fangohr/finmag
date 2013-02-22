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
import dolfin as df
from finmag.util.timings import default_timer
from finmag.energies.demag.solver_gcr import FemBemGCRSolver
from finmag.energies.demag.solver_fk import FemBemFKSolver
from copy import deepcopy

#Generate all possible solver parameters
def solver_parameters(solver_exclude, preconditioner_exclude):
    linear_solver_set = ["lu"] 
    linear_solver_set += [e[0] for e in df.krylov_solver_methods()]
    preconditioner_set = [e[0] for e in df.krylov_solver_preconditioners()] #FIXME: not used

###################################################################
#To the user:
###################################################################

#This command prints out possible linalg solver parameters
##df.info(df.LinearVariationalSolver.default_parameters(), 1)

class LinAlgDemagTester(object):
    """
    A class to test the speed of linalg solvers and preconditioners
    used in demag calculation. 
    """
    def __init__(self,solver,testparams,mesh,m, degree=1, element="CG",
                 project_method='magpar', unit_length=1,Ms = 1.0):
        """
        
        *Arguments*
            solver
                Demag solver type, "GCR" or "FK".

            testparams
                A list of dictionaries containing parameters for the linear solve.
                         Can also give a tuple of two dictionaries if different parameters are
                         wished for the two linear solves.
            mesh
                dolfin Mesh object
            m
                the Dolfin object representing the (unit) magnetisation
            Ms
                the saturation magnetisation 
            parameters
                dolfin.Parameters of method and preconditioner to linear solvers.
            degree
                polynomial degree of the function space
            element
                finite element type, default is "CG" or Lagrange polynomial.
            unit_length
                the scale of the mesh, defaults to 1.
            project_method
                possible methods are
                    * 'magpar'
                    * 'project'
            bench
                set to True to run a benchmark of linear solvers
        Note: Running groups of testparams instead of individual params saves on BEM assembly.

        """
        self.testparams = testparams
        print mesh
        if solver == "FK":
            self.solver = FemBemFKSolver(mesh,m, degree = degree,
                                         element=element,
                                         project_method = project_method,
                                         unit_length = unit_length,Ms = Ms)
        elif solver == "GCR":
            self.solver = FemBemGCRSolver(mesh,m, degree = degree,
                                          element=element,
                                          project_method = project_method,
                                          unit_length = unit_length,Ms = Ms)
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

            #Set the linear solver parameters
            if self.solver.__name__ == "GCR Demag Solver":
                matrix = self.solver.poisson_matrix_dirichlet
            else:
                matrix = self.solver.poisson_matrix
            self.solver.poisson_solver = df.KrylovSolver(matrix,
                                                         t1["linear_solver"],
                                                         t1["preconditioner"])
            self.solver.laplace_solver = df.KrylovSolver(self.solver.poisson_matrix,
                                                         t2["linear_solver"],
                                                         t2["preconditioner"]) 
            #Get the timings
            self.solver.solve()
            self.report(t1,t2)
                          
            #Copy the timer and reset the old one.
            self.timelist.append(deepcopy(default_timer))
            default_timer.reset()
                          
        #After the testing is finished delete the BEM to free up memory.
        del self.solver.bem

    def report(self,t1,t2,n = 10):
        print "".join(["Linear solve timings of the ",self.solver.__name__])
        print "mesh size in verticies = ",self.solver.mesh.num_vertices()
        print "First solve parameters"
        print t1
        print "Second solve parameters"
        print t2
        print "\n",default_timer.report(n)

if __name__ == "__main__":
    #Solver type for use in the script, "FK" or "GCR".
    fembemsolvertype = "FK"

    ##Default linear solver parameters to test.
    ## Each entry can either be a single dictionary or a tuple/list of dictionaries.
    ## In the single case both linear solves will be computed using the specified parameters.
    ## If a tuple/list is given the 1st solve is done according to the 1st parameter dictionary,
    ## and the 2nd solve according to the 2ns parameter dictionary.

    default_params =[{"linear_solver":"tfqmr",'preconditioner':"ilu"}, \
                     {"linear_solver":"tfqmr",'preconditioner':"default"},    \
                     {"linear_solver":"bicgstab",'preconditioner':"ilu"}, \
                     {"linear_solver":"bicgstab",'preconditioner':"default"},
                     {"linear_solver":"tfqmr",'preconditioner':"bjacobi"}]

#[1;37;34mrichardson, none: 0.003942 s#[0m
#[1;37;34mcg, none: 0.004244 s#[0m
#[1;37;34mbicgstab, none: 0.005020 s#[0m
#[1;37;34mgmres, none: 0.005216 s#[0m

    #have we swapped preconditioners and linear_solvers? Not sure (HF).
    #GB The above dictionary is correct, ilu is a preconditioner and cg is a krylov solver.
    
    #to use all methods, activate next line
    #default_params = solver_parameters(solver_exclude=[], preconditioner_exclude=[])
    #print default_params


    #As a default plot a sequence of solver values for FK with different meshes
    
    import finmag.tests.demag.problems.prob_fembem_testcases as pft
    import matplotlib.pyplot as plt

    #Create a range of mesh sizes
    sizelist = [4,3]
    #Important use floats here so the entry order stays consistent 
    sizelist = [0.7,0.6,0.5]
    problems = [pft.MagSphereBase(radius=10, maxh=i) for i in sizelist]

    #Run the tests
    testers = [LinAlgDemagTester(fembemsolvertype,default_params,p.mesh,p.m) for p in problems]
    for t in testers:
        t.test()
        
    meshsizes = numpy.array([p.mesh.num_vertices() for p in problems])
    
    #Output
    plt.grid(True)
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
