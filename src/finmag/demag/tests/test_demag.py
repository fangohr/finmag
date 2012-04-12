import pytest
import dolfin as df
import numpy as np
#from finmag.demag import solver_fk, solver_gcr
from finmag.demag.solver_fk_test import SimpleFKSolver
from finmag.demag.problems.prob_fembem_testcases import MagSphere

TOL = 3e-1

problems = [MagSphere(5,3), MagSphere(5,2), MagSphere(5,1)]

solvers = [SimpleFKSolver]

fields = []
cases = []
for problem in problems:
    for solver in solvers:
        case = solver(problem.V, problem.m, problem.Ms)
        grad = case.compute_field()
        cases.append(case)
        fields.append(grad)



@pytest.mark.parametrize(("case", "grad"), zip(cases, fields))
def test_one_third(case, grad):
    """Demag field should be one third of initial magnetisation."""
    W = df.VectorFunctionSpace(case.V.mesh(), "CG", 1, dim=3)
    a = df.interpolate(df.Expression(problem.M), W)

    #Note to self check the array's 
    assert len(grad) == len(a.vector().array())

    diff =  np.abs(grad + 1./3*a.vector().array())
    print "Max difference should be zero, is %g." % max(diff)
    assert max(diff) < TOL



@pytest.mark.parametrize(("case", "grad"), zip(cases, fields))
def test_y_z_zero(case, grad):
    """Demag field should be zero in y- and z-direction."""
    y_start = len(grad)//3
    y = grad[y_start:2*y_start]
    z = grad[2*y_start:]
    maxy = max(abs(y))
    print "Max field in y-direction should be zero, is %g." % maxy
    assert maxy < TOL

    maxz = max(abs(z))
    print "Max field in z-direction should be zero, is %g." % maxz
    assert maxz < TOL



@pytest.mark.parametrize(("case", "grad"), zip(cases, fields))
def test_x_deviation(case, grad):
    """Deviation of demag field in x-direction should be zero."""
    x = grad[:len(grad)//3]
    dev = abs(max(x) - min(x))
    
    print "Max deviation in x-direction should be zero, is %g." % dev
    assert dev < TOL



@pytest.mark.parametrize(("case", "grad"), zip(cases, fields))
def test_norm_x(case, grad):
    """Norm of demag field in x-direction minus 1/3 Ms should be zero."""
    x = grad[:len(grad)//3] #+ 1./3*case.Ms
    print len(x), len(grad)
    
    y = grad.reshape((
    return
    normx = np.linalg.norm(x)

    print "L2 norm of the x-component + 1/3 Ms should be zero. Is %g." % normx
    assert normx < TOL



@pytest.mark.parametrize(("case", "grad"), zip(cases, fields))
def test_avg_x(case, grad):
    """Average of the x-components should be 1/3 of Ms."""
    x = grad[:len(grad)//3]
    avg = np.average(x) 

    print "Average of x-component should be -%g. Is %g." % (1./3*problem.Ms, avg)
    assert abs(avg+1./3*problem.Ms) < TOL


if __name__ == '__main__':
    print ""
    i = 0
    for case, grad in zip(cases, fields):
        print "Test case %d: %s" % (i+1, problems[i].desc())
        print "Number of vertices:", case.V.mesh().num_vertices()
        #test_one_third(case, grad)
        #test_y_z_zero(case, grad)
        #test_x_deviation(case, grad)
        test_norm_x(case, grad)
        #test_avg_x(case, grad)
        print ""
        i += 1
