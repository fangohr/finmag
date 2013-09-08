from __future__ import division
import numpy as np
import dolfin as df
#import pytest
import os
#import finmag
#from finmag.energies import Zeeman, TimeZeeman, DiscreteTimeZeeman, OscillatingZeeman
from finmag.energies import Zeeman
#from finmag.util.consts import mu0
from finmag.util.meshes import pair_of_disks
from finmag.util.helpers import vector_valued_function
#from math import sqrt, pi, cos, sin


def test_energies_in_subdomains(tmpdir):
    """
    Create a mesh with two subdomains. For each energy class compute the energy
    on each subdomain and compare with the total energy on the whole mesh. Also
    compare with analytical expressions if feasible.

    """
    os.chdir(str(tmpdir))

    # Create a mesh consisting of two disks (with different heights)
    d = 30.0
    h1 = 5.0
    h2 = 10.0
    sep = 10.0
    maxh = 2.5
    Ms = 8.6e5
    unit_length = 1e-9
    RTOL = 5e-3  # achievable relative tolerance depends on maxh

    mesh = pair_of_disks(d, d, h1, h2, sep, theta=0, maxh=maxh)
    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)

    # Create a mesh function for the two domains (each representing one disk),
    # where the regions are marked with '0' (first disk) and '1' (second disk).
    class Disk1(df.SubDomain):
        def inside(self, pt, on_boundary):
            x, y, z = pt
            return np.linalg.norm([x, y]) < 0.5 * (d + sep)

    class Disk2(df.SubDomain):
        def inside(self, pt, on_boundary):
            x, y, z = pt
            return np.linalg.norm([x, y, z]) > 0.5 * (d + sep)

    disk1 = Disk1()
    disk2 = Disk2()
    domains = df.CellFunction("size_t", mesh)
    domains.set_all(0)
    disk1.mark(domains, 1)
    disk2.mark(domains, 2)
    dx = df.Measure("dx")[domains]
    dx_disk_1 = dx(1)
    dx_disk_2 = dx(2)

    def m_init_1(pt):
        return [1, 0, 0]

    def m_init_2(pt):
        return [0.5, -0.8, 0]

    def m_init(pt):
        x, y, z = pt
        if np.linalg.norm([x, y]) < 0.5 * (d + sep):
            res = m_init_1(pt)
        else:
            res = m_init_2(pt)
        return res

    m = vector_valued_function(m_init, V, normalise=True)

    interaction = Zeeman(1e6 * np.array([1, 0, 0]))
    interaction.setup(V, m, Ms, unit_length=unit_length)

    E_computed_1 = interaction.compute_energy(dx=dx_disk_1)
    E_computed_2 = interaction.compute_energy(dx=dx_disk_2)
    E_computed_total = interaction.compute_energy(dx=df.dx)

    # Check that the sum of the computed energies for disk #1 and #2 equals the total computed energy
    assert np.allclose(E_computed_1 + E_computed_2, E_computed_total, atol=0, rtol=1e-12)
