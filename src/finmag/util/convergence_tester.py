"""
Test the convergence of various solvers, using various problems and various
norms. The results can be outputted as a print statement report or as plots
using matplotlib. Plots are organised by error norm and function in one plot,
with each solver having a sequence of points representing the solutions
to the problems.

#Note the code is little abstract and under development
#User friendliness will be included eventually
"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt


class ConvergenceTester(object):
    """Converge tester using problems,solvers, and norms"""
    def __init__(self,test_solver_classes,reference_solver_class,test_solutions,problems,norms,xaxis,cases = None):
        """
        Explanation of parameters:
        obligatory
            test_solver_classes    - Dictionary of solvernames:solverclasses. solverclasses will solve the problems
            reference_solver_class - Dictionary with a solvername:solverclass. The test solvers will be compared to this one
            test_solutions         - Dictionary with solutionname:attribute. The attributes will be fetched from all of the solutions
                                     after solve() is called.
            problems               - Dictionary with problemname:problemobject. The problemobjects will be given to the solvers 
            norms                  - Dictionary with normname: normfunction. The normfunction must accept two dolfin functions
                                     and return a number which should be some sort of measure of error.
            xaxis                  - Tuple with axisname: List of numbers to plot the errors against.
        Optional
            cases                  - List of tuples(function,norm) which specifies which functions should be measured against the
                                     reference solution in which norms.
        """
        
        self.test_solver_classes = test_solver_classes
        self.reference_solver_class = reference_solver_class
        self.test_solutions = test_solutions
        self.problems = problems
        self.norms = norms
        self.xaxis = xaxis
        self.cases = cases
        #A dictionary with various styles of points for plotting in matplotlib
        self.styledic = {0:'D',1:'p',2:',',3:'o',4:'v',5:'^',6:'<',7:'1',8:'2',9:'3',\
                         10:'4',11:'s',12:'.',13:'*',14:'h',15:'H',16:'+',17:'x',\
                         18:'-',19:'d',20:'|'} 

        self.__main()

    def print_report(self):
        """Command line Report of the results"""
        starline = "*****************************************************"
        print starline
        print "Convergence Report"
        print starline
        print self.xaxis[0] #Should print the name of the xaxis we are testing against, for example num finite elements
        print self.xaxis[1] #Plots the x axis data

        #For every kind of solver...
        for solvername in self.errordata:
            print starline
            print "Convergence of ", solvername
            print starline
            #plot every kind of error
            for errorname in self.errordata[solvername]:
                print errorname
                print self.errordata[solvername][errorname]
                print

    def __main(self):
        refsolvername = self.reference_solver_class.keys()[0]
        refsolver = self.reference_solver_class[refsolvername]
        
        #Solve all the problems with the test solvers
        self.test_solverdata = {k:self.__get_solver_data(self.test_solver_classes[k],self.problems) for k in self.test_solver_classes}
        
        #Solve all the problems with the reference solver
        self.ref_solverdata = {refsolvername:self.__get_solver_data(refsolver,self.problems)}
                                     
        #Compute errors of functions compared to the reference solution.
        self.errordata = {solvername:self.__get_error_data(self.test_solverdata[solvername],\
                          self.ref_solverdata[refsolvername]) for solvername in self.test_solverdata}

    def __get_solver_data(self,solverclass,problems):
        """
        Takes a Solver class and a list of problems
        and returns a dictionary of solution functions,
        with the values being a list of solutions
        """
        #Create the solver objects with the problems
        solverobjs = [solverclass(p) for p in self.problems]

        #Solve all the problems
        dummy = [s.solve() for s in solverobjs]
        
        #Extract and store the various solutions
        #Value of test_solutions should be the function we want to take from a solverobject
        solutions = {solname: [getattr(s,self.test_solutions[solname]) for s in solverobjs] for solname in self.test_solutions }
        
        return solutions

    def __get_error_data(self,sd1,sd2):
        """
        Calculate the various errors that we are interested in,
        comparing the functions in sd1 to those of sd2.
        sd = solver_data.
        """

        if self.cases is None:
            #Return a dictionary with all possible norm-function combinations
            return {normname + " " + solname : self.__get_convergence_data(sd1[solname],sd2[solname], self.norms[normname]) \
                    for solname in test_solutions for normname in self.norms}
        else:
            #Return only the norm-function combinations from self.cases        
            return {solname + " " + normname: self.__get_convergence_data(sd1[solname],sd2[solname], self.norms[normname]) for solname,normname in self.cases}

    def __get_convergence_data(self,fn_test, fn_ref, norm):
        """
        Test the convergence of a sequence of solutions
        to some sequence of analytical values in a norm

        fn_approx = list of dolfin functions
        fn_ana = list of dolfin functions
        norm = function that takes two dolfin functions as arguments
        """
        tups = list(zip(fn_test, fn_ref))
        tups.reverse()
        errors = [norm(s,a) for s,a in tups]
        return errors

    def plot_results(self):
        """Output the convergence data using matplotlib"""
        assert len(self.test_solver_classes) <= len(self.styledic), \
               "Can only plot " +str(len(self.styledic)) + "or less solvers, due to \
                a limited number of plotting point styles available"

        #This specificies a subplot grid 2x3
        plotgrid = "22"
        #Generate the plots
        figurenum = 2
        #For every error-norm combination we want to plot...
        for i,errorname in enumerate(self.errordata[self.errordata.keys()[0]]):
            if i <> 0 and i % 4 ==0:
                #The subplots are full so start a new page
                plt.figure(figurenum)
                figurenum += 1
            #put the subplot in position i modulo 4 + 1
            plotnum = plotgrid + str(i%4+1)
            plt.subplot(*plotnum)
            
            #... the solutions errors for evey solver
            for j,sol in enumerate(self.test_solver_classes):
                #TODO plot the error data correctly
                plt.plot(self.xaxis[1],self.errordata[sol][errorname],self.styledic[j],label = sol)
            #Give the plot titles
            plt.title(errorname)
            plt.xlabel(self.xaxis[0])
            plt.ylabel("Error")
            plt.legend()
        plt.show()

if __name__ == "__main__":
###Output a Convergence report for FK and GCR solvers###

    from finmag.demag import solver_fk, solver_gcr
    import finmag.util.error_norms as en
    from finmag.demag.problems import prob_fembem_testcases as pft
    import finmag.demag.solver_base as sb

    
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
            self.get_demagfield()
            #Split the Demagfield into Component functions
            self.Hdemagx,self.Hdemagy,self.Hdemagz = self.Hdemag.split(True) 
        
        def get_demagfield(self):
            self.Hdemag = project(Expression(("1.0/3.0","0.0","0.0")),self.Hdemagspace)

    #Extended versions of the solvers that give us some extra functions
    class FemBemFKSolverTest(solver_fk.FemBemFKSolver):
        """Extended verions of FemBemFKSolver used for testing in 3d"""
        def solve(self):
            super(FemBemFKSolverTest,self).solve()
            #Split the Demagfield into Component functions
            self.Hdemagx,self.Hdemagy,self.Hdemagz = self.Hdemag.split(True) 

    class  FemBemGCRSolverTest(solver_gcr.FemBemGCRSolver):
        """Extended verions of  FemBemGCRSolver used for testing in 3d"""
        def solve(self):
            super(FemBemGCRSolverTest,self).solve()
            #Split the Demagfield into Component functions
            self.Hdemagx,self.Hdemagy,self.Hdemagz = self.Hdemag.split(True) 
    
    finenesslist = range(2,4)    
    problems = [pft.MagUnitSphere(n) for n in finenesslist]

    #Xaxis
    numelement = [p.mesh.num_cells() for p in problems]
    xaxis = ("Number of elements",numelement)

    #Solvers
    test_solver_classes = {"FK Solver": FemBemFKSolverTest,"GCR Solver": FemBemGCRSolverTest}
    reference_solver_class = {"Analytical":FemBemAnalytical}

    #Test solutions
    test_solutions = {"Phi":"phi","Hdemag":"Hdemag","Hdemag X":"Hdemagx",\
                     "Hdemag Y":"Hdemagy","Hdemag Z":"Hdemagz"}
    
    #Norms
    norms = {"L2 Error":en.L2_error,"Discrete Max Error":en.discrete_max_error}

    #cases
    cases = [("Phi","L2 Error"),("Phi","Discrete Max Error"),("Hdemag","L2 Error"), \
             ("Hdemag X","Discrete Max Error"),("Hdemag Y","Discrete Max Error"),\
             ("Hdemag Z","Discrete Max Error")]

    #Create a ConvergenceTester and generate a report
    ct = ConvergenceTester(test_solver_classes,reference_solver_class,test_solutions,problems,norms,xaxis,cases)
    ct.print_report()
    ct.plot_results()
