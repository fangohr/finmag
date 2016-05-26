# This script attempts to solve an unphysical Temperature (T)-based dolfin
# problem within an array-based integrator. It initialises a gaussian, and
# decays all values independently.
#
# Written for dolfin 1.6.0.
#
# Run with mpirun -n 2 python array_intergrator_parallel.py.

import dolfin as df
import dolfinh5tools
import integrators
import numpy as np
import sys


# For parallelness, get rank.
rank = df.mpi_comm_world().Get_rank()

# Specifying the initial problem.
mesh = df.IntervalMesh(20, -1, 1)
funcSpace = df.FunctionSpace(mesh, 'CG', 1)
initState = df.Expression("exp(-(pow(x[0], 2) / 0.2))")  # Gaussian
initFuncVal = df.interpolate(initState, funcSpace)
initialArray = initFuncVal.vector().array()

# Gather the initial array
initRecv = df.Vector()
initFuncVal.vector().gather(initRecv, np.array(range(funcSpace.dim()), "intc"))

# Print stuff.
print("{}: My vector is of shape {}.".format(rank, initialArray.shape[0]))
print("{}: My array looks like:\n {}.".format(rank, initialArray))
print("{}: My mesh.coordinates are:\n {}."
      .format(rank, funcSpace.mesh().coordinates()))
print("{}: The initial gathered array looks like:\n {}."
      .format(rank, initRecv.array()))

# Defining behaviour in time using dolfin.
def dTt_dolfin(T, t):
    """
    Finds dT/dt using dolfin.

    Arguments:
       T: Array representing the temperature at a specific time.
       t: Single value of time at which to find the derivative.
    Returns:
       The derivative of T with respect to t as an array.
    """

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
        T = integrators.euler(T, dTt_dolfin(T, t), tStep)
    return T


print("{}: Integrating...".format(rank))
tEnd = 1
T = run_until(tEnd, initFuncVal.vector().array())

print("{}: My vector is of shape {}.".format(rank, len(T)))
print("{}: My array looks like:\n {}.".format(rank, T))

# Create function space (and by extension, a vector object) for a fancy Dolfin
# gathering operation, so we can plot data.
TSend = df.Function(funcSpace)
TSend.vector()[:] = T

TRecv = df.Vector()
TSend.vector().gather(TRecv, np.array(range(funcSpace.dim()), "intc"))
print("{}: The gathered array looks like:\n {}.".format(rank, TRecv.array()))

# Plot the curves. This should look bizarre, as the gather reconstructs the
# data in the incorrect order.
if rank == 0:
    import matplotlib.pyplot as plt
    plt.plot(initRecv.array())
    plt.plot(TRecv.array())
    plt.show()
    plt.close()

# Save this data. This stores data in the correct order intelligently.
sd = dolfinh5tools.lib.openh5("array_integrator_parallel", funcSpace, mode="w")
sd.save_mesh()
sd.write(initFuncVal, "T", 0)
sd.write(TSend, "T", tEnd)
sd.close()

# Now that you've got to here, you can run the script
# "load_array_integrator_parallel_data.py" to plot the data in the correct
# order, using the data we have just saved.
