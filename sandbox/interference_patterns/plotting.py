# I have some trouble plotting a function f(x, y) = z on a surface,
# when all I have are vectors for x, y and z.

import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# Let's say this is the data I got from the simulation.
nx = 3
ny = 2
mesh = df.Rectangle(0, 0, 1, 1, nx, ny)
nodes = mesh.num_vertices()
x = mesh.coordinates()[:,0]
y = mesh.coordinates()[:,1]
z = np.random.rand(nodes) # that's what I want to plot z = f(x, y)
assert x.shape == y.shape == z.shape == (nodes, )

# That's how numpy would like the data.
xnp, ynp = np.meshgrid(np.linspace(0, 1, nx+1), np.linspace(0, 1, ny+1))

print "Dolfin:\n", x, "\n", y
print "Matplotlib:\n", xnp, "\n", ynp

# Now modify shape for matplotlib.

xx = x.view().reshape((ny+1, -1))
yy = y.view().reshape((ny+1, -1))
zz = z.view().reshape((ny+1, -1))

print "Modified Dolfin:\n", xx, "\n", yy, "\nvalues:\n", zz
print "The plot shows {} squares. We had {} values for z.".format(nx*ny, nodes)

assert xx.shape == yy.shape == zz.shape
plt.pcolor(xx, yy, zz)
plt.colorbar()
plt.show()
