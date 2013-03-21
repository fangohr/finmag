import dolfin as df
from finmag.energies.demag.fk_demag import FKDemag
from finmag.util.meshes import sphere
from simple_timer import SimpleTimer

# nmag, dict ksp_tolerances in Simulation class constructor
# apparently gets passed to petsc http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPSetTolerances.html
# dolfin
# managed to pass tolerances to KrylovSolver, may want to simply use "solve" instead
# https://answers.launchpad.net/dolfin/+question/193119
# check /usr/share/dolfin/demo
# check help(df.solve)

mesh = sphere(r=10.0, maxh=0.4)
S3 = df.VectorFunctionSpace(mesh, "CG", 1)
m = df.Function(S3)
m.assign(df.Constant((1, 0, 0)))
Ms = 1
unit_length = 1e-9

demag = FKDemag()
demag.setup(S3, m, Ms, unit_length)

benchmark = SimpleTimer()
print "Default df.KrylovSolver tolerances (abs, rel): Poisson ({}, {}), Laplace ({}, {}).".format(
        demag._poisson_solver.parameters["absolute_tolerance"],
        demag._poisson_solver.parameters["relative_tolerance"],
        demag._laplace_solver.parameters["absolute_tolerance"],
        demag._laplace_solver.parameters["relative_tolerance"])
with benchmark:
    for i in xrange(10):
        print demag.compute_field().reshape((3, -1)).mean(axis=1)
print "With default parameters {} s.".format(benchmark.elapsed)

demag._poisson_solver.parameters["absolute_tolerance"] = 1e-5
demag._laplace_solver.parameters["absolute_tolerance"] = 1e-5
demag._poisson_solver.parameters["relative_tolerance"] = 1e-5
demag._laplace_solver.parameters["relative_tolerance"] = 1e-5
with benchmark:
    for i in xrange(10):
        print demag.compute_field().reshape((3, -1)).mean(axis=1)
print "With higher tolerances {} s.".format(benchmark.elapsed)
