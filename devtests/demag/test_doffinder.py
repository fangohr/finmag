#Test the dofinder module
from doffinder import *
from prob_testcases import *


def test_doffinder1d():
    problem = MagUnitInterval()
    degree = 1
    V = FunctionSpace(problem.mesh,"CG",degree)

    bounddofset = bounddofs(V,degree, problem.coreboundfunc,2)
    assert len(bounddofset) == 2,"Failure in doffinder module, a 1-d demagproblem should only have 2 boundary facets"
                                  
