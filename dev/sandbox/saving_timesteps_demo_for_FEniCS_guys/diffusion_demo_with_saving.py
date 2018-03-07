"""
Extension of the FEniCS tutorial demo for the diffusion equation
with Dirichlet conditions.
"""

from dolfin import *
import numpy
from time import sleep
import tables as pytables

# Create mesh and define function space
nx = ny = 20
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary conditions
alpha = 3; beta = 1.2
u0 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                alpha=alpha, beta=beta, t=0)

class Boundary(SubDomain):  # define the Dirichlet boundary
    def inside(self, x, on_boundary):
        return on_boundary

boundary = Boundary()
bc = DirichletBC(V, u0, boundary)

# Initial condition
u_1 = interpolate(u0, V)
uuu = interpolate(u0, V)
uuu.rename('uuu', 'UUU')

dt = 0.3      # time step

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(beta - 2 - 2*alpha)
a = u*v*dx + dt*inner(nabla_grad(u), nabla_grad(v))*dx
L = (u_1 + dt*f)*v*dx

A = assemble(a)   # assemble only once, before the time stepping
b = None          # necessary for memory saving assemeble call

u = Function(V)   # the unknown at a new time level
T = 1.9           # total simulation time
t = dt


# For now we also write to xdmf and hdf5 files to see how dolfin does
# this at the moment, and to be able to compare with our proposed
# format. This will disappear in the final demo.
F = XDMFFile('solution_xdmf.xdmf')
G = HDF5File('solution_hdf5.h5', 'w')
F << (u_1, 0.0)
G.write(u_1.vector(), 'myvector_0')

# Prepare for saving to HDF5 file (using pytables)
class VectorDescription(pytables.IsDescription):
    """
    Customized record for writing solution vectors including a counter
    and time information.
    """
    counter = pytables.Int64Col()     # Signed 64-bit integer
    values  = pytables.FloatCol(shape=u_1.vector().array().shape)   # 
    time    = pytables.FloatCol()      # double (double-precision)

h5file = pytables.openFile("diffusion_data.h5", mode='w')
group = h5file.createGroup('/', 'Vector')
table = h5file.createTable(group, 'myvector', VectorDescription)
tabdata = table.row


# Run the simulation
counter = 0
while t <= T:
    print 'time =', t
    b = assemble(L, tensor=b)
    u0.t = t
    bc.apply(A, b)
    solve(A, u.vector(), b)

    # Verify
    u_e = interpolate(u0, V)
    maxdiff = numpy.abs(u_e.vector().array() - u.vector().array()).max()
    print 'Max error, t=%.2f: %-10.3f' % (t, maxdiff)

    t += dt
    u_1.assign(u)

    # Write to xdmf and hdf5 using dolfin's API
    F << (u_1, t)
    F << (uuu, t)
    G.write(u_1.vector(), 'myvector_{}'.format(counter))

    # Write data in our own format using pytables
    tabdata['counter'] = counter
    tabdata['values'] = u_1.vector()
    tabdata['time']  = t
    tabdata.append()
    counter += 1

# Close (and flush) the file
h5file.close()
