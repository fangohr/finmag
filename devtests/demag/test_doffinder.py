#Test the dofinder module
from doffinder import bounddofs
from prob_trunc_testcases import *


def test_doffinder1d():
    problem = MagUnitInterval()
    degree = 1
    V = FunctionSpace(problem.mesh,"CG",degree)

    bounddofset = bounddofs(V,degree, problem.coreboundfunc,2)
    print bounddofset
    assert len(bounddofset) == 2,"Failure in doffinder module, a 1-d demagproblem should only have 2 boundary facets"
                                  

if __name__ == "__main__":
    test_doffinder1d()
