from test_bem_computation import compute_scalar_potential_native_fk
from finmag.util.meshes import sphere
import numpy as np
from mayavi import mlab

# Create the mesh and compute the scalar potential
mesh = sphere(r=1.0, maxh=0.1)
phi = compute_scalar_potential_native_fk(mesh)

x, y, z = mesh.coordinates().T
values = phi.vector().array()

# Set up the visualisation
figure = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
# Use unconnected datapoints and scalar_scatter and interpolate using a delaunay mesh
src = mlab.pipeline.scalar_scatter(x, y, z, values)
field = mlab.pipeline.delaunay3d(src)
# Make the contour plot
contours = np.linspace(np.min(values), np.max(values), 10)
mlab.pipeline.iso_surface(field, contours=list(contours))

mlab.show()
