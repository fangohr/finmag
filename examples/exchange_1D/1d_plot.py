import numpy
import finmag.util.helpers as helpers
from mayavi import mlab

"""
Visualise the final configuration of the magnetisation.

"""

x = numpy.genfromtxt("1d_coord.txt")
y = numpy.zeros(len(x))
z = numpy.zeros(len(x))

Ms = numpy.genfromtxt("1d_M.txt")
Mx, My, Mz = helpers.components(Ms[-1])

figure = mlab.figure(bgcolor=(0,0,0), fgcolor=(1,1,1))
q = mlab.quiver3d(x, y, z, Mx, My, Mz, figure=figure)
q.scene.z_plus_view()
mlab.axes(figure=figure)

mlab.show()


