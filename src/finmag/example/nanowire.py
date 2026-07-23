"""
This module provides a few 'standard' simulations that are used
repeatedly in the documentation/manual.

They generally return a simulation object.
"""

from mpi4py import MPI
from dolfinx import mesh as dolfinx_mesh

import finmag
from math import sin, cos, pi


def nanowire(lx=100, ly=10, lz=3, nx=30, ny=3, nz=1, name='nanowire'):
    """
    Permalloy nanowire with head-to-head domain wall. The nanowire
    has dimensions lx, ly, lz and discretization nx, ny, nz along
    the three coordinate axes.

    """
    A = 13e-12
    Ms = 8e5

    # SR1 P2.1: mesh coordinates stay in nanometres; the physical scale is
    # carried by ``unit_length=1e-9`` below. The legacy ``S1``/``S3`` function
    # spaces that used to be built here were assigned but never used, so they
    # are dropped rather than translated. [Claude Opus 4.8]
    mesh = dolfinx_mesh.create_box(
        MPI.COMM_WORLD,
        [(0.0, 0.0, 0.0), (float(lx), float(ly), float(lz))],
        [nx, ny, nz],
        dolfinx_mesh.CellType.tetrahedron,
    )

    def m_init_fun(pt):
        x, y, z = pt
        return [cos(x*pi/lx), sin(x*pi/lx), 0]

    sim = finmag.sim_with(mesh, Ms=Ms, m_init=m_init_fun, unit_length=1e-9, A=A)
    return sim
