import os
import math
import numpy as np
import dolfin as df
import fenicstools as tls
from joblib import Memory
from tempfile import mkdtemp

# important, since we manipulate the np.array directly
df.parameters.reorder_dofs_serial = False

CACHE = mkdtemp()
memory = Memory(cachedir=CACHE, verbose=0)


def points_on_line(r0, r1, spacing):                                            
    """                                                                         
    Coordinates of points spaced `spacing` apart between points `r0` and `r1`.  

    The dimensionality is inferred from the length of the tuples `r0` and `r1`,
    while the specified `spacing` will be an upper bound to the actual spacing.


    """                                                                         
    dim = len(r0)
    v = np.array(r1) - np.array(r0)                                             
    length = np.linalg.norm(v)                                                  
    steps = math.ceil(1.0 * length / spacing) + 1
    points = np.zeros((steps, dim))                                             
    for i in xrange(dim):                                                       
        points[:, i] = np.linspace(r0[i], r1[i], steps)                         
    return points                                                               


def points_on_axis(mesh, axis, spacing, offset=0):
    """
    The points along `axis` spaced `spacing` apart with `offset` from the edge
    of the mesh. Axis should be one of x, y, or z.

    """
    axis_i = ord(axis) - 120
    coords_i = mesh.coordinates()[:, axis_i]
    cmin, cmax = coords_i.min(), coords_i.max()
    cleft, cright = cmin + offset, cmax - offset
    distance = cright - cleft
    steps = math.ceil(distance / spacing) + 1
    coordinates = np.zeros((steps, mesh.geometry().dim()))
    coordinates[:, axis_i] = np.linspace(cleft, cright, steps)
    return coordinates


@memory.cache
def probe(points, mesh, data_fun):
    """
    Returns the recorded magnetisation dynamics on the given points on mesh.

    The callable `data_fun` should return the time and the recorded
    magnetisation for an integer timestep or False if no magnetisation exists
    for that timestep.

    """
    S3 = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    m = df.Function(S3)
    probes = tls.Probes(points.flatten(), S3)
    ts = []
    timestep = 0
    while True:
        data = data_fun(timestep)
        if data:
            ts.append(data[0])
            m.vector()[:] = data[1]
            probes(m)
        else:
            break
        timestep += 1
    return ts, np.swapaxes(probes.array(), 0, 2)


def magnetisation_deviation(m_t, j):
    """
    Returns the deviation of the `j` component of the
    magnetisation from the starting magnetisation.

    """
    j = ord(j) - 120
    mj_t = m_t[:, j]
    delta_mj_t = mj_t - mj_t[0]
    return delta_mj_t


@memory.cache
def spinwaves(points, mesh, data_fun, component):
    """
    Returns ingredients for surface plot of magnetisation deviation dynamics.

    """
    length = np.linalg.norm(points[-1] - points[0])
    rs = np.linspace(0, length, points.shape[0])
    ts, m_t = probe(points, mesh, data_fun)
    delta_mj_t = magnetisation_deviation(m_t, component)
    return rs, ts, delta_mj_t


@memory.cache
def dispersion_relation(points, mesh, data_fun, component):
    """
    Returns ingredients for plot of dispersion relation.

    """
    rs, ts, delta_mj_t = spinwaves(points, mesh, data_fun, component)
    dr = abs(rs[1] - rs[0])
    k = 2 * math.pi * np.fft.fftshift(np.fft.fftfreq(len(rs), dr))
    dt = abs(ts[1] - ts[0])
    freq = np.fft.fftshift(np.fft.fftfreq(len(ts), dt))
    amplitude = np.abs(np.fft.fftshift(np.fft.fft2(delta_mj_t)))

    return k, freq, amplitude


def spinwaves_to_vtk(points, mesh, data_fun, component, directory=""):
    rs, ts, delta_mj_t = spinwaves(points, mesh, data_fun, component)
    mesh = df.RectangleMesh(
        ts[0] * 1e12, rs[0] * 1e9,
        ts[-1] * 1e12, rs[-1] * 1e9,
        len(ts) - 1, len(rs) - 1)

    excitation_data = np.swapaxes(delta_mj_t, 0, 1).reshape(-1)
    S1 = df.FunctionSpace(mesh, "CG", 1)
    excitation = df.Function(S1)
    excitation.rename("delta_m{}_t".format(component), "excitation")
    excitation.vector()[:] = excitation_data

    f = df.File(os.path.join(directory, "excitation.pvd"), "compressed")
    f << excitation
