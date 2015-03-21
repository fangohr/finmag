"""
This module provides a few 'standard' simulations that are used
repeatedly in the documentation/manual.

They generally return a simulation object.
"""

import dolfin as df
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

    mesh = df.BoxMesh(0, 0, 0, lx, ly, lz, nx, ny, nz)
    S1 = df.FunctionSpace(mesh, 'CG', 1)
    S3 = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)

    def m_init_fun(pt):
        x, y, z = pt
        return [cos(x*pi/lx), sin(x*pi/lx), 0]

    sim = finmag.sim_with(mesh, Ms=Ms, m_init=m_init_fun, unit_length=1e-9, A=A)
    return sim
