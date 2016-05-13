# This script attempts to solve a Temperature-based diffusion-like dolfin
# problem within an array-based integrator. Look for the balloons <!>.

import dolfin as df


# Specifying the initial problem.
# mesh = df.IntervalMesh(100, -10, 10)
mesh = df.UnitSquareMesh(100, 100)
# funcSpace = df.VectorFunctionSpace(mesh, 'CG', 1)
funcSpace = df.FunctionSpace(mesh, 'CG', 1)
initState = df.Expression("exp(-(pow(x[0], 2) / 0.2))")  # Gaussian
initFuncVal = df.interpolate(initState, funcSpace)

initialArray = initFuncVal.vector().array()
print("My vector is of shape {}.".format(initialArray.shape[0]))
print("My array looks like:\n {}.".format(initialArray))


# Defining behaviour in time.
def dTt(T):
    """
    Finds dT/dt (eventually).

    Arguments:
       T: Array representing the temperature at a specific time.
       funcSpace: Dolfin function space to interpolate T to.
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
    # describes the mathematical function that calcultes dT/dt from T.
    f = df.grad(TOld) #df.inner(df.grad(TOld), df.grad(TOld)) # <!> Failure here?

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

# <!>
# df.plot(initFuncVal)
# df.plot(dTt(initFuncVal.vector().array()))

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
        T = euler(T, dTt(T), tStep)
    return T


print("Integrating...")
T = run_until(10, initFuncVal.vector().array())

print("My vector is of shape {}.".format(T))
print("My array looks like:\n {}.".format(T))

# Gather result arrays.
# from mpi4py import MPI as mpi
# comm = mpi.COMM_WORLD

# # Plot the two curves.
# import matplotlib.pyplot as plt
# plt.plot(initialArray)
# plt.plot(T)
# plt.show()
# plt.close()

# Updating the array in this way updates the dolfin function?
# print("My vector is of shape {}.".format(initFuncVal))
