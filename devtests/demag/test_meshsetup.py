#A set of tests to insure that the submesh and boundary generation of
#Interiorboundary and  TruncDemagProblem

from dolfin import *
from prob_testcases import *
import math

#Global Tolerance for inexact comparisons in percent
TOL = 0.02
class TestProblems(object):

    def test_1d(self):
        problem = MagUnitInterval()
        #Test to see if the Volume is correct
        vol = self.bound_volume(problem)
        assert near(vol,2.0), "Error in 1-d internal boundary creation, approximate volume %g does not equal 2"%(vol)
        #Test to see if the number of facets is correct
        cfound = problem.corebound.countfacets
        cactual = self.bound_facets(problem)
        assert cfound == cactual, "Error in 1-d internal boundary creation, the number of facets in the generated boundary \
                                  %d does not equal that of the coremesh boundary %d"%(cfound,cactual)

    def test_2d(self):
        problem = MagUnitCircle()
        #Test to see if the Volume is correct
        vol = self.bound_volume(problem)
        print vol
        voltrue = 2*problem.r*math.pi
        print self.compare(vol,voltrue)
        assert self.compare(vol,voltrue), "Error in 2-d internal boundary creation, error in approximate volume %g is not within TOL %g of \
                                      the true volume %g"%(vol,TOL,voltrue)
##        #Test to see if the number of facets is correct
##        cfound = problem.corebound.countfacets
##        cactual = self.bound_facets(problem)
##        assert cfound == cactual, "Error in 2-d internal boundary creation, the number of facets in the generated boundary \
##                                  %d does not equal that of the coremesh boundary %d"%(cfound,cactual)
        
    def bound_volume(self,problem):
        #Gives the volume of the surface of the magnetic core
        V = FunctionSpace(problem.mesh,"CG",1)
        one = interpolate(Constant(1),V)
        volform = one('-')*problem.dSC
        return assemble(volform,interior_facet_domains = problem.coreboundfunc) 

    def bound_facets(self,problem):
        #Gives the number of facets in the boundary of the coremesh
        boundmesh = BoundaryMesh(problem.coremesh)
        return boundmesh.num_cells()

    def compare(self,est,trueval):
        #Returns if the error in the estimate less than TOL percent of trueval
        relerror = abs((est - trueval)/trueval )
        print relerror
        return relerror < TOL
test = TestProblems()
test.test_2d()
