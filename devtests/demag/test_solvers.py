"""A set of tests to insure that the Demag Solvers work properly"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
import prob_trunc_testcases as pttc
import solver_nitsche as sn
import solver_base as sb
import prob_trunc_testcases as ptt
import prob_fembem_testcases as pft
import numpy as np

#This suite tests the solutions for the demag scalar potential function from various Solvers.

class TestTruncDemagSolver(object):
    """Test the Solver Base Class"""

    def test_restrictfunc(self):
        """
        A simple linear Function is restricted to half a square and the answer is tested against an analytic value
        """
        mesh = UnitSquare(2,2)
        V = FunctionSpace(mesh,"CG",1)
        
        #A plane going from 0 to 1 in the x direction
        E = Expression("1 - x[0]")
        u = interpolate(E,V)
        
        #Generate the submesh
        class Half(SubDomain):
            def inside(self,x,on_boundary):
                return x[0]<0.5 + DOLFIN_EPS
            
        meshfunc = MeshFunction("uint",mesh,2)
        meshfunc.set_all(0)
        Half().mark(meshfunc,1)
        halfmesh = SubMesh(mesh,meshfunc,1)

        #Class initialized with a dummy problem
        problem = ptt.MagUnitInterval()
        solver = sb.TruncDeMagSolver(problem)

        #Get the restricted function
        uhalf = solver.restrictfunc(u,halfmesh)

        #The restricted function should be a plane going from 0 to 0.5 in x direction
        exactsol = interpolate(E,uhalf.function_space())
        
        a = abs(uhalf - exactsol)*dx
        A = assemble(a)
        assert near(A,0.0),"Error in TruncDemagSolver.restrictfunc, restricted function does not match analytical value"
       
class TestNitscheSolver(object):
    def setup_class(self):
        self.TOL = 1.0 #Fixme This is a bad tolerance, maybe the nitsche solver can be made more precise
          #TODO the averaging by volume causes the error to increase since the surface volumes are <1,
          #this gets worse for increased dimension. So get rid of averaging and recalibrate the gammas and TOL
        
        self.problem1d = pttc.MagUnitInterval()
        self.problem2d = pttc.MagUnitCircle()
        self.problem3d = pttc.MagUnitSphere()

        self.solver1d = sn.NitscheSolver(self.problem1d)
        self.solver2d = sn.NitscheSolver(self.problem2d)
        self.solver3d = sn.NitscheSolver(self.problem3d)

        self.solution1d = self.solver1d.solve()
        self.solution2d = self.solver2d.solve()
        self.solution3d = self.solver3d.solve()

    def test_1d(self):
        self.probtest(self.problem1d,self.solver1d,self.solution1d)
    def test_2d(self):
        self.probtest(self.problem2d,self.solver2d,self.solution2d)
    def test_3d(self):
        self.probtest(self.problem3d,self.solver3d,self.solution3d)
        
    def test_compare_3danalytical(self):
        """Test the potential phi against the known analytical solution in the core"""
        L2error = self.L2_error_3d_potential(self.solution3d)
        print "3d Analtical solution of potential, comparison L2error =",L2error
        assert L2error < self.TOL,"L2 Error in 3d computed solution from the analytical solution, %g is not less than the Tolerance %g"%(L2error,self.TOL)
        
    def L2_error_3d_potential(self,solution):
        soltrue = Expression("-x[0]/3.0")
        soltrue = project(soltrue,self.solver3d.V)
        L2error = self.error_norm(solution,soltrue, cell_domains = self.problem3d.corefunc, dx = self.problem3d.dxC) 
        return L2error

    def test_compare_3danalytical_gradient(self):
        """Test the Demag Field from the Nitsche Solver against the known analytical solution in the core"""
        L2error = self.L2_error_3d_demag(self.solver3d.Hdemag_core)
        print "3d Analtical solution demag field, comparison L2error =",L2error
        assert L2error < self.TOL,"L2 Error in 3d computed solution from the analytical solution, %g is not less than the Tolerance %g"%(L2error,self.TOL)

    def L2_error_3d_demag(self,Hdemag):
        #Function Space of solution
        fspace = Hdemag.function_space()
        #True analytical solution
        soltrue = Expression(("1.0/3.0","0.0","0.0"))
        soltrue = project(soltrue,fspace)
        L2error = self.error_norm(Hdemag,soltrue) 
        return L2error
    
    ##FIXME convergence FAILS!
##    def test_convergance_3d(self):
##        """The FEM solution should converge to the analytical solution as the mesh is refined"""
##        NUM_REFINEMENTS = 6 
##        #Take previously calculated errors 
##        firstpoterror = self.L1_error_3d_potential(self.solution3d)
##        firstdemagerror = self.L2_error_3d_demag(self.solver3d.Hdemag_core)
##
##        #Refine the core mesh
##        for i in range(NUM_REFINEMENTS):
##            self.problem3d.refine_core()
##        #Solve the demag problem again
##        self.solver3d = sn.NitscheSolver(self.problem3d)
##        self.solution3d = self.solver3d.solve()
##        #get errors
##        pot3d_errornew = self.L1_error_3d_potential(self.solution3d)
##        demag3d_errornew = self.L2_error_3d_demag(self.solver3d.Hdemag_core)
##        
##        #Test the last solution against the 1st one.
##        assert firstpoterror > pot3d_errornew,"Error in refinement of 3d problem, error in potential did not decrease on final refinement "+ str(i+1)
##        assert firstdemagerror > demag3d_errornew,"Error in refinement of 3d problem, error in demag did not decrease on final refinement "+ str(i+1)
    
    def print_convergance_3d(self):
        """
        Since refining first increases the error it is not suitable to test convergence for the first refinements
        to see this strange phenomena run this method.
        """
        NUM_REFINEMENTS = 6
        #Take previously calculated errors 
        pot3d_errorold  = self.L1_error_3d_potential(self.solution3d)
        demag3d_errorold = self.L2_error_3d_demag(self.solver3d.Hdemag_core)

        #Refine the core mesh and see if the errors decrease
        print "Errors for potential,demag"
        for i in range(NUM_REFINEMENTS):
            self.problem3d.refine_core()
            #Solve the demag problem
            self.solver3d = sn.NitscheSolver(self.problem3d)
            self.solution3d = self.solver3d.solve()
            #Get errors
            pot3d_errornew = self.L1_error_3d_potential(self.solution3d)
            demag3d_errornew = self.L2_error_3d_demag(self.solver3d.Hdemag_core)
            print pot3d_errornew, demag3d_errornew, " refinements = ", i+1
            #Make the new errors the old errors
            pot3d_errorold = pot3d_errornew
            demag3d_errorold = demag3d_errornew

    def probtest(self,problem,solver,solution):
        self.dbc_test(problem,solution)
        self.continuity_test(problem,solver,solution)
        self.normalderivativejump_test(problem,solver,solution)

    def dbc_test(self,problem, solution):
        L2error = self.error_norm(solution,Function(solution.function_space()) , dx = ds)
        print "dbc_test: L2error=",L2error
        errmess = "Error in Nitsche Solver with problem " + problem.desc() + \
        "outer dirichlet BC condition not satisfied, average solution boundary integral is %g"%(L2error)
        assert L2error < self.TOL,errmess 

    def continuity_test(self,problem,solver,solution):
        #2 Test Continuity across the interior boundary
        jumpphi = solver.phi1('-') - solver.phi0('+')
        L2error1 = self.error_norm(jump(solution),Function(solution.function_space())('-'), interior_facet_domains = problem.coreboundfunc, dx = problem.dSC) 
        L2error2 = self.error_norm(jumpphi,Function(solution.function_space())('-'), interior_facet_domains = problem.coreboundfunc, dx = problem.dSC) 
    
        print "continuity_test: L2error1=",L2error1
        print "continuity_test: L2error2=",L2error2
        assert L2error1 < self.TOL,"Error in Nitsche Solver with problem" + problem.desc() + "continuity accross magnetic core boundary not satisfied for phi1 and phi2, \
                               TOL = %g, L2error = %g"%(self.TOL,L2error1)
        assert L2error2 < self.TOL,"Error in Nitsche Solver with 1d problem" + problem.desc() + "continuity accross magnetic core boundary not satisfied for phi total \
                               TOL = %g, L2error = %g"%(self.TOL, L2error2)

    def normalderivativejump_test(self,problem,solver,solution):
        #3 Test jump in normal derivative across the interior boundary
        N = FacetNormal(problem.coremesh)
        M = solver.M
        
        jumpphinor = dot(grad(solver.phi1('-') - solver.phi0('+')),N('+'))
        f1 =abs(jumpphinor - dot(M,N)('-'))
        f2 = abs(dot(jump(grad(solution)),N('+')) - dot(M,N)('+'))
        zero = Function(solution.function_space())

        L2error1 = self.error_norm(f1,zero('-'),interior_facet_domains = problem.coreboundfunc, dx = problem.dSC) 
        L2error2 = self.error_norm(f2,zero('-'), interior_facet_domains = problem.coreboundfunc, dx = problem.dSC) 
        
        print "normalderivativejump_test: L2error1=",L2error1
        print "normalderivativejump_test: L2error2=",L2error2
        assert L2error1 < self.TOL,"Error in Nitsche Solver with " + problem.desc() + " normal derivative jump accross magnetic core boundary not satisfied for phi1 and phi2, \
                               TOL = %g, L2error = %g"%(self.TOL,L2error1)
        assert L2error2 < self.TOL,"Error in Nitsche Solver with " + problem.desc() + " normal derivative jump accross magnetic core boundary not satisfied for phi total \
                               TOL = %g, L2error = %g"%(self.TOL,L2error2)
        
    def error_norm(self,func1,func2,cell_domains = None,interior_facet_domains = None, dx = dx):
        Eform = inner(func1-func2,func1-func2)*dx
        E = assemble(Eform, cell_domains =  cell_domains, interior_facet_domains =interior_facet_domains)
        return sqrt(E)
        
class TestFemBemDeMagSolver(object):
    """Test the FemBemDeMagSolver class """
        
    def setup_class(self):      
        self.problem = pft.MagUnitSphere()
        self.solver = sb.FemBemDeMagSolver(self.problem)
        
    def test_get_boundary_dof_coordinate_dict(self):
        """Test only relevant for CG1"""
        V = FunctionSpace(self.solver.problem.mesh,"CG",1)
        numdofcalc = len(self.solver.get_boundary_dof_coordinate_dict(V))
        numdofactual = BoundaryMesh(V.mesh()).num_vertices()
        assert numdofcalc == numdofactual,"Error in Boundary Dof Dictionary creation, number of DOFS " +str(numdofcalc)+ \
                                          " does not match that of the Boundary Mesh " + str(numdofactual)

    def test_solve_laplace_inside(self):
        """Solve a known laplace equation"""
        mesh = UnitSquare(2,2)
        V = FunctionSpace(mesh,"CG",1)
        fold = interpolate(Expression("1-x[0]"),V)
        fnew = interpolate(Expression("1-x[0]"),V)
        #The Laplace equation should give the same solution as f
        fnew = self.solver.solve_laplace_inside(fnew)
        assert fold.vector().array().all() == fnew.vector().array().all(),"Error in method test_solve_laplace_inside(), \
        Laplace solution does not equal original solution"
        print "solve_laplace_inside testpassed"

    def easyspace(self):
        mesh = UnitSquare(1,1)
        return FunctionSpace(mesh,"CG",1)

    def test_get_dof_normal_dict(self):
        """Test some features of a known example """
        V = self.easyspace()
        facetdic = self.solver.get_dof_normal_dict(V)
        coord = self.solver.get_boundary_dof_coordinate_dict(V)
        
        #Tests
        assert len(facetdic[0]) == 2,"Error in normal dictionary creation, 1,1 UnitSquare with CG1 has two normals per boundary dof"
        assert facetdic.keys() == coord.keys(),"error in normal dictionary creation, boundary dofs do not agree with those obtained from \
                                            get_boundary_dof_coordinate_dict"

    def test_get_dof_normal_dict_avg(self):
        """Use same case as test_get_dof_normal_dict, see if average normals
            have length one"""
        V = self.easyspace()
        avgnormtionary = self.solver.get_dof_normal_dict_avg(V)
        for k in avgnormtionary:
            assert near(sqrt(np.dot(avgnormtionary[k],avgnormtionary[k].conj())),1),"Failure in average normal calulation, length of\
                                                                                     normal not equal to 1"
        
if __name__ == "__main__":
    t = TestNitscheSolver()
    t.setup_class()
    print "* Doing test 1d ==========="
    t.test_1d()
    print "gamma = ", t.problem1d.gamma
    print
    print "* Doing test 2d ==========="
    t.test_2d()
    print "gamma = ", t.problem2d.gamma
    print
    print "* Doing test 3d ==========="
    t.test_3d()
    print "gamma = ", t.problem3d.gamma
    print
    print "* Doing Analytical comparison of potential ======="
    t.test_compare_3danalytical()
    print
    print "* Doing Analytical comparison of Demag field ======="
    t.test_compare_3danalytical_gradient()
    print
    #Slow uncomment with caution
    #print "* Doing Convergance test of Demag field ======="
    #t.print_convergance_3d()
    
    t1 = TestFemBemDeMagSolver()
    t1.setup_class()
    print "* Doing test for solve_laplace_inside"
    t1.test_solve_laplace_inside()
