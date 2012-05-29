#A set of tests to insure that the submesh and boundary generation of
#Interiorboundary and  TruncDemagProblem

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
import finmag.demag.problems.prob_trunc_testcases as pttc
import math

#Global Tolerance for inexact comparisons in percent
TOL = 0.01
class TestTruncMeshSetup(object):
    def test_1d(self):
        problem = pttc.MagUnitInterval()
        self.problem_tests(problem)

    def test_2d(self):
        problem = pttc.MagUnitCircle()
        self.problem_tests(problem)

    def test_3d(self):
        problem = pttc.MagUnitSphere()
        self.problem_tests(problem)

    def problem_tests(self,problem):
        #Test to see if the internal boundary volume is correct
        vol = self.bound_volume(problem)
        voltrue = self.submesh_volume(problem)
        assert self.compare(vol,voltrue), "Error in internal boundary creation for problem " + problem.desc() + \
                                          " error in approximate volume %g is not within TOL %g of the true volume %g"%(vol,TOL,voltrue)
        #Test to see if the number of internal boundary facets is correct
        cfound = problem.corebound.countfacets
        cactual = self.bound_facets(problem)
        assert cfound == cactual, "Error in internal boundary creation for problem " + problem.desc() + \
                                   " the number of facets in the generated boundary %d does not equal that of the coremesh boundary %d"%(cfound,cactual)
        #Test to see if the core mesh refinement works
        self.refinement_test(problem)

    def bound_volume(self,problem):
        """Gives the volume of the surface of the magnetic core"""
        V = FunctionSpace(problem.mesh,"CG",1)
        one = interpolate(Constant(1),V)
        volform = one('-')*problem.dSC
        return assemble(volform,interior_facet_domains = problem.coreboundfunc) 

    def submesh_volume(self,problem):
        """coreboundmesh = BoundaryMesh(problem.coremesh)"""
        V = FunctionSpace(problem.coremesh,"CG",1)
        one = interpolate(Constant(1),V)
        volform = one*ds
        return assemble(volform)
    
    def bound_facets(self,problem):
        """Gives the number of facets in the boundary of the coremesh"""
        boundmesh = BoundaryMesh(problem.coremesh)
        return boundmesh.num_cells()

    def compare(self,est,trueval):
        """Returns if the error in the estimate less than TOL percent of trueval"""
        relerror = abs((est - trueval)/trueval )
        print relerror
        return relerror < TOL

    def refinement_test(self,problem):
        """Test to see if the core refinement works"""
        numcellsbefore =problem.mesh.num_cells()
        problem.refine_core()
        assert problem.mesh.num_cells() > numcellsbefore,"Error in core mesh refinement in problem " + problem.desc()+  " total number of cells did not increase"
