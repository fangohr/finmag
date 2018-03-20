from dolfin import *
from time import time
import math, random
#from finmag.util.meshes import from_geofile

def run_test():
    solver = KrylovSolver("cg", "jacobi")

    mesh = Box(0,0,0,30,30,100,10,10,30)
    #mesh = Mesh(convert_mesh("bar.geo"))
    #mesh = UnitCubeMesh(32,32,32)
    V = FunctionSpace(mesh, "CG", 1)
    W = VectorFunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)
    w = TrialFunction(W)

    A = assemble(inner(grad(u), grad(v))*dx)

    D = assemble(inner(w, grad(v))*dx)
    m = Function(W)
    m.vector()[:] = random.random()
    b = D*m.vector()
    x = Vector()
    start = time()
    solver.solve(A, x, b)
    return time() - start

parameters["linear_algebra_backend"] = "PETSc"
time1 = run_test()
parameters["linear_algebra_backend"] = "PETScCusp"
time2 = run_test()
print "Backend: PETSc, time: %g" % time1
print "Backend: PETScCusp, time: %g" % time2

print "Speedup: %g" % (time1/time2)
