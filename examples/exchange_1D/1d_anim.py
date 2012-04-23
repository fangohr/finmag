import numpy
import finmag.sim.helpers as helpers
from mayavi import mlab

"""
Visualise the evolution of the vector field M over time.

"""

x = numpy.genfromtxt("1d_coord.txt")
y = numpy.zeros(len(x))
z = numpy.zeros(len(x))

Ms = numpy.genfromtxt("1d_M.txt")
u, v, w = helpers.components(Ms[0])

figure = mlab.figure(bgcolor=(0,0,0), fgcolor=(1,1,1))
q = mlab.quiver3d(x, y, z, u, v, w, figure=figure)
q.scene.z_plus_view()
mlab.axes(figure=figure)

it = 0
@mlab.animate(delay=1000)
def animation():
    global it
    while True:
        u, v, w = helpers.components(Ms[it])
        q.mlab_source.set(u=u, v=v, w=w, scalars=w)
        it += 1
        if it == len(Ms):
            print "End of data."
            break
        yield
anim = animation()
mlab.show()
