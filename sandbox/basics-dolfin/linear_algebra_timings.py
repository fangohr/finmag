__author__ = "Anders Logg <logg@simula.no>"
__date__ = "2012-01-20"
__copyright__ = "Copyright (C) 2012 Anders Logg"
__license__  = "GNU LGPL version 3 or any later version"

# Last changed: 2012-01-20

from dolfin import *
import pylab

# Mesh sizes to check
mesh_sizes = [2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91]

# Global data for plotting
_legends = []
_markers = "ov^<>1234sp*hH+xDd|_"

def create_linear_system(n):
    "Create linear system for Poisson's equation on n x n x n mesh"

    mesh = UnitCubeMesh(n, n, n)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression("sin(5*x[0])*sin(5*x[1])*sin(5*x[2])")
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    A = assemble(a)
    b = assemble(L)

    bc = DirichletBC(V, 0.0, DomainBoundary())
    bc.apply(A, b)

    x = Vector(b.size())

    return A, x, b

def bench_solver(solver, preconditioner="default"):
    "Benchmark given solver and preconditioner"

    global _legends
    global _markers

    print "Computing timings for (%s, %s)..." % (solver, preconditioner)

    # Compute timings
    sizes = []
    timings = []
    for n in mesh_sizes:
        A, x, b = create_linear_system(n)
        N = b.size()
        if N > 20000 and solver in ("lu", "cholesky"): break
        sizes.append(N)
        print "  Solving linear system of size %d x %d" % (N, N)
        tic()
        solve(A, x, b, solver, preconditioner)
        timings.append(toc())

    # Plot timings
    marker = _markers[len(_legends) % len(_markers)]
    pylab.loglog(sizes, timings, "-%s" % marker)

    # Store legend
    backend = parameters["linear_algebra_backend"]
    if preconditioner == "default":
        _legends.append("%s %s" % (backend, solver))
    else:
        _legends.append("%s %s, %s" % (backend, solver, preconditioner))

# Timings for uBLAS
parameters["linear_algebra_backend"] = "uBLAS"
bench_solver("lu")
bench_solver("cholesky")
bench_solver("gmres", "ilu")

# Timings for PETSc
parameters["linear_algebra_backend"] = "PETSc"
bench_solver("lu")
bench_solver("gmres", "none")
bench_solver("gmres", "ilu")
bench_solver("cg", "none")
bench_solver("cg", "ilu")
bench_solver("gmres", "amg")
bench_solver("cg", "amg")
bench_solver("tfqmr", "ilu")

# Finish plot
pylab.grid(True)
pylab.title("Solving Poisson's equation with DOLFIN 1.0.0")
pylab.xlabel("N")
pylab.ylabel("CPU time")
pylab.legend(_legends, "upper left")
pylab.savefig("linear-algebra-timings.pdf")
pylab.savefig("linear-algebra-timings.png")
print("Data plotted in linear-algebra-timings.png/pdf")
pylab.show()
