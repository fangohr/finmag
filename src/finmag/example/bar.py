"""
This module provides a few 'standard' simulations that are used
repeatedly in the documentation/manual.

They generally return a simulation object.
"""

import dolfin as df
import finmag


def bar(name='bar', demag_solver_type=None):
    """Py bar with dimensions 30x30x100nm, initial field
    pointing in (1,0,1) direction.

    Same as example 2 in Nmag manual.

    This function returns a simulation object that is 'ready to go'.

    Useful commands to run this for a minute::

        times = numpy.linspace(0, 3.0e-11, 6 + 1)
        for t in times:
            # Integrate
            sim.run_until(t)
    """

    xmin, ymin, zmin = 0, 0, 0      # one corner of cuboid
    xmax, ymax, zmax = 30, 30, 100  # other corner of cuboid
    nx, ny, nz = 15, 15, 50         # number of subdivisions (use ~2nm edgelength)
    mesh = df.BoxMesh(xmin, ymin, zmin, xmax, ymax, zmax, nx, ny, nz)

    sim = finmag.sim_with(mesh, Ms=0.86e6, alpha=0.5, unit_length=1e-9,
                A=13e-12, m_init=(1, 0, 1), name=name,
                demag_solver_type=demag_solver_type)

    return sim


def barmini(name='barmini', mark_regions=False, demag_solver_type=None):
    """Py bar with dimensions 3x3x10nm, initial field
    pointing in (1,0,1) direction.

    Same as example 2 in Nmag manual, but much smaller (and faster).

    This function returns a simulation object that is 'ready to go'.

    If `mark_regions` is True, the mesh will be subdivided vertically
    into two regions which are markes as 'top' and 'bottom'.

    Useful commands to run this for a couple of seconds:

        times = numpy.linspace(0, 3.0e-11, 6 + 1)
        for t in times:
            # Integrate
            sim.run_until(t)
    """

    xmin, ymin, zmin = 0, 0, 0      # one corner of cuboid
    xmax, ymax, zmax = 3, 3, 10     # other corner of cuboid
    nx, ny, nz = 2, 2, 4            # number of subdivisions (use ~2nm edgelength)
    mesh = df.BoxMesh(xmin, ymin, zmin, xmax, ymax, zmax, nx, ny, nz)

    sim = finmag.sim_with(mesh, Ms=0.86e6, alpha=0.5, unit_length=1e-9,
                A=13e-12, m_init=(1, 0, 1), name=name,
                demag_solver_type=demag_solver_type)

    if mark_regions:
        def fun_regions(pt):
            return 'bottom' if (pt[2] <=5.0) else 'top'
        sim.mark_regions(fun_regions)

    return sim
