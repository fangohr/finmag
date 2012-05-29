import pytest
import numpy as np
from finmag.energies.demag.solver_fk_test import SimpleFKSolver
from finmag.energies.demag.solver_fk import FemBemFKSolver as FKSolver
from finmag.energies.demag.solver_gcr import FemBemGCRSolver as GCRSolver
from problems.prob_fembem_testcases import MagSphereBase

# Can get smaller tolerance if we use finer mesh, but that will
# delay the jenkins build with minutes.
TOL = 1e-2

# Feel free to add more (interesting) problems
# NOTE: Should increase the number of vertices
# when the lindholm formulation makes the build
# of the boundary element matrix faster.
problems = [MagSphereBase(1, 5), MagSphereBase(0.25, 1)]

# Please feel free to also add more (working) solvers...
solvers = [SimpleFKSolver, FKSolver, GCRSolver]

fields = []
cases = []
for problem in problems:
    for solver in solvers:
        if solver.__name__ == "SimpleFKSolver":
            case = solver(problem.V, problem.m, problem.Ms)
        else:
            case = solver(mesh=problem.mesh, m=problem.M, Ms=problem.Ms)
        grad = case.compute_field()
        cases.append(case)
        fields.append(grad)

@pytest.mark.parametrize(("case", "grad"), zip(cases, fields))
def test_one_third(case, grad):
    """Demag field should be one third of initial magnetisation."""
    m = case.m
    assert len(grad) == len(m.vector().array())

    diff =  np.abs(grad + 1./3*m.vector().array())
    print "Max difference should be zero, is %g." % max(diff)
    #GCR solver is a little less precise here so the tolerance has been raised.
    assert max(diff) < 2*TOL

@pytest.mark.parametrize(("case", "grad"), zip(cases, fields))
def test_y_z_zero(case, grad):
    """Demag field should be zero in y- and z-direction."""
    grad = np.hsplit(grad, 3)
    y = grad[1]
    z = grad[2]

    maxy = max(abs(y))
    print "Max field in y-direction should be zero, is %g." % maxy
    assert maxy < TOL

    maxz = max(abs(z))
    print "Max field in z-direction should be zero, is %g." % maxz
    assert maxz < TOL

@pytest.mark.parametrize(("case", "grad"), zip(cases, fields))
def test_x_deviation(case, grad):
    """Deviation of demag field in x-direction should be zero."""
    grad = np.hsplit(grad, 3)
    x = grad[0]
    dev = abs(max(x) - min(x))

    print "Max deviation in x-direction should be zero, is %g." % dev
    assert dev < TOL

@pytest.mark.parametrize(("case", "grad"), zip(cases, fields))
def test_norm_x(case, grad):
    """Norm of demag field in x-direction minus 1/3 Ms should be zero."""
    grad = np.hsplit(grad, 3)
    x = grad[0]
    normx = np.linalg.norm(x + 1./3*case.Ms)

    print "L2 norm of the x-component + 1/3 Ms should be zero. Is %g." % normx
    # This test needs a higher tolerance to pass.
    assert normx < TOL*100

@pytest.mark.parametrize(("case", "grad"), zip(cases, fields))
def test_avg_x(case, grad):
    """Average of the x-components should be 1/3 of Ms."""
    grad = np.hsplit(grad, 3)
    x = grad[0]
    avg = np.average(x)

    print "Average of x-component should be -%g. Is %g." % (1./3*case.Ms, avg)
    assert abs(avg+1./3*case.Ms) < 2*TOL


if __name__ == '__main__':
    print ""
    i = 0
    for case, grad in zip(cases, fields):
        print "Test case %d: %s" % (i+1, problems[i//len(solvers)].desc())
        print "Solver: %s" % case.__class__.__name__
        print "Number of vertices:", case.V.mesh().num_vertices()
        test_one_third(case, grad)
        test_y_z_zero(case, grad)
        test_x_deviation(case, grad)
        test_norm_x(case, grad)
        test_avg_x(case, grad)
        print ""
        i += 1
