# This script attempts to solve a Temperature-based diffusion-like dolfin
# problem within an array-based integrator.
#
# Run with mpirun -n 2 python array_intergrator_parallel.py.

import dolfin as df
import numpy as np

# For parallelness, get rank.
rank = df.mpi_comm_world().Get_rank()

# Specifying the initial problem.
mesh = df.IntervalMesh(20, -1, 1)
# mesh = df.UnitSquareMesh(100, 100)
# funcSpace = df.VectorFunctionSpace(mesh, 'CG', 1)
funcSpace = df.FunctionSpace(mesh, 'CG', 1)
initState = df.Expression("exp(-(pow(x[0], 2) / 0.2))")  # Gaussian
# initState = df.Constant(rank)
initFuncVal = df.interpolate(initState, funcSpace)

initialArray = initFuncVal.vector().array()
print("{}: My vector is of shape {}.".format(rank, initialArray.shape[0]))
print("{}: My array looks like:\n {}.".format(rank, initialArray))
print("{}: My mesh.coordinates are:\n {}.".format(rank, mesh.coordinates()))

# Gather the initial array
initRecv = df.Vector()
initFuncVal.vector().gather(initRecv, np.array(range(funcSpace.dim()), "intc"))
print("{}: The initial gathered array looks like:\n {}."
      .format(rank, initRecv.array()))

# Defining behaviour in time using dolfin.
def dTt_dolfin(T):
    """
    Finds dT/dt using dolfin.

    Arguments:
       T: Array representing the temperature at a specific time.
       funcSpace: Dolfin function space to interpolate T to.
    Returns:
       The derivative of T with respect to t as an array.
    """
    # import ipdb; ipdb.set_trace()

    # Convert T to dolfin function from array.
    TOld = df.Function(funcSpace)
    TOld.vector()[:] = T

    # Solve the "linear" problem to find dT/dt.

    # This 'a' represents the unknown, which contains no new information, but
    # will eventually contribute to the solution later.
    TNew = df.TrialFunction(funcSpace)
    v = df.TestFunction(funcSpace)
    a = TNew * v * df.dx

    # 'f' here represents an expression (but not a dolfin expression) which
    # describes the mathematical function that calculates dT/dt from T.
    f = TOld * df.Constant(-0.9) # df.inner(df.grad(TOld), df.grad(TOld)) # <!> Failure here?

    # This 'L' represents what we know, and will be used to calculate our
    # solution eventually.
    L = f * v
    L *= df.dx

    # This is not actually the solution, but it is where the solution will end
    # up eventually, once the solver has done its work.
    solutionEventually = df.Function(funcSpace)

    # The problem defines what we want to know, what we do know, and where to
    # put the solution. The solution argument is not actually the solution
    # (yet), but it's where the solution will end up eventually.
    problem = df.LinearVariationalProblem(a, L, solutionEventually)

    # The solver solves the problem eventually.
    solver = df.LinearVariationalSolver(problem)

    # Now we solve the problem. solutionEventually is now populated with the
    # solution to the problem.
    solver.solve()

    # Convert and return our solution.
    return solutionEventually.vector().array()

    # Calculate derivative dT/dx (and by extension dT/dt).
    # dTdx = df.inner(df.grad(TOld), df.grad(TOld))
    # dTdt = dTdx * df.Constant(0.1)
    # outFunc = df.dot(dTdt * df.interpolate(df.Expression(["1."]), funcSpace),
    #                  df.Expression(["1."]))
    # dTdx = df.grad(TOld)[0, 0]
    # dTdt = df.project(dTdx * 0.1, funcSpace)
    # return -0.1 * T

    # Convert and return the derivative dT/dt.
    # return outFunc.vector().array()


# Defining behaviour in time, clumsily. This behaviour is replicated by
# dTt_dolfin.
# def dTt(T):
#     """
#     Finds dT/dt clumsily.

#     This represents an unphysical linear decay.

#     Arguments:
#        T: Array representing the temperature at a specific time.
#        funcSpace: Dolfin function space to interpolate T to.
#     Returns:
#        The derivative of T with respect to t as an array.

#     """

#     return T * -0.9


# Euler
def euler(Tn, dTndt, tStep):
    """
    Performs Euler integration to obtain T_{n+1}.

    Arguments:
       Tn: Array-like representing Temperature at time t_n.
       dTndt: Array-like representing dT/dt at time t_n.
       tStep: Float determining the time to step over.

    Returns T_{n+1} as an array-like.
    """
    return Tn + dTndt * tStep


def run_until(t, T0, steps=100):
    """
    Integrates the problem for time t.

    Arguments:
       t: Float determining total time to integrate over.
       T0: Array denoting initial temperature.
       steps: Integer number of integration steps to perform over the time t.
    Returns integrated quantity as an array.
    """

    tStep = t / float(steps)
    T = T0  # Initial Temperature
    for step in xrange(int(steps)):
        # T = euler(T, dTt_dolfin(T), tStep)
        T = euler(T, dTt_dolfin(T), tStep)
    return T


print("{}: Integrating...".format(rank))
T = run_until(1, initFuncVal.vector().array())

print("{}: My vector is of shape {}.".format(rank, len(T)))
print("{}: My array looks like:\n {}.".format(rank, T))

# Create function space (and by extension, a vector object) for a fancy Dolfin
# gathering operation.
TSend = df.Function(funcSpace)
TSend.vector()[:] = T

TRecv = df.Vector()
TSend.vector().gather(TRecv, np.array(range(funcSpace.dim()), "intc"))
print("{}: The gathered array looks like:\n {}.".format(rank, TRecv.array()))

# Plot the curves.
if rank == 0:
    import matplotlib.pyplot as plt
    plt.plot(initRecv.array())
    plt.plot(TRecv.array())
    plt.show()
    plt.close()