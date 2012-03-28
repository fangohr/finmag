import pytest
import dolfin as df
import numpy as np
from finmag.demag import solver_fk, solver_gcr
from finmag.demag.problems import prob_fembem_testcases as pft

TOL = 1e-3

problems = [pft.MagUnitSphere(n) for n in (1,2)] 
#problems.append(pft.MagSphere10())
solvers = [solver_fk.FemBemFKSolver, solver_gcr.FemBemGCRSolver]

fields = []
cases = []
for problem in problems:
    for solver in solvers:
        case = solver(problem)
        phi = case.solve()
        grad = case.get_demagfield(phi)
        cases.append(case)
        fields.append(grad)


@pytest.mark.xfail
@pytest.mark.parametrize(("case", "grad"), zip(cases, fields))
def test_one_third(case, grad):
    """Demag field should be one third of initial magnetisation."""
    W = df.VectorFunctionSpace(case.problem.mesh, "DG", 0, dim=3)
    a = df.interpolate(df.Constant(case.problem.M), W)

    #Note to self check the array's 
    diff =  np.abs(grad.vector().array() - 1./3*a.vector().array())
    print "Max difference should be zero, is %g." % max(diff)
    assert max(diff) < TOL


@pytest.mark.xfail
@pytest.mark.parametrize(("case", "grad"), zip(cases, fields))
def test_y_z_zero(case, grad):
    """Demag field should be zero in y- and z-direction."""
    y, z = grad.split(True)[1:]
    y, z = y.vector().array(), z.vector().array()
    
    maxy = max(abs(y))
    print "Max field in y-direction should be zero, is %g." % maxy
    assert maxy < TOL

    maxz = max(abs(z))
    print "Max field in z-direction should be zero, is %g." % maxz
    assert maxz < TOL


@pytest.mark.xfail
@pytest.mark.parametrize(("case", "grad"), zip(cases, fields))
def test_x_deviation(case, grad):
    """Deviation of demag field in x-direction should be zero."""
    x = grad.split(True)[0].vector().array()
    dev = abs(max(x) - min(x))
    
    print "Max deviation in x-direction should be zero, is %g." % dev
    assert dev < TOL


@pytest.mark.xfail
@pytest.mark.parametrize(("case", "grad"), zip(cases, fields))
def test_norm_x(case, grad):
    """Norm of demag field in x-direction minus 1/3 Ms should be zero."""
    x = grad.split(True)[0].vector().array() - 1./3*case.problem.Ms
    normx = np.linalg.norm(x)

    print "L2 norm of the x-component - 1/3 Ms should be zero. Is %g." % normx
    assert normx < TOL


@pytest.mark.xfail
@pytest.mark.parametrize(("case", "grad"), zip(cases, fields))
def test_avg_x(case, grad):
    """Average of the x-components should be 1/3 of Ms."""
    x = grad.split(True)[0].vector().array()
    avg = np.average(x) 

    print "Average of x-component should be %g. Is %g." % (1./3*problem.Ms, avg)
    assert abs(avg-1./3*Ms) < TOL


if __name__ == '__main__':
    for case, grad in zip(cases, fields):

        test_one_third(case, grad)
        print
        test_y_z_zero(case, grad)
        print
        test_x_deviation(case, grad)
        print
        test_norm_x(case, grad)
        print
        test_avg_x(case, grad)

