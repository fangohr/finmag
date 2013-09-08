from __future__ import division
import numpy as np
import dolfin as df
import pytest
import os
import finmag
#from finmag.energies import Zeeman, TimeZeeman, DiscreteTimeZeeman, OscillatingZeeman
from finmag.energies import Zeeman
#from finmag.util.consts import mu0
from finmag.util.meshes import pair_of_disks
from finmag.util.helpers import vector_valued_function
#from math import sqrt, pi, cos, sin


class MultiDomainTest(object):
    def __init__(self, mesh, get_domain_id, m_vals, Ms, unit_length=1e-9):
        """
        `get_domain_id` is a function of the form (x, y, z) -> id which maps
        some point coordinates in the mesh to an integer identifying the domain
        which the point belongs to.

        """
        self.mesh = mesh
        self.get_domain_id = get_domain_id
        self.domain_ids = [get_domain_id(pt) for pt in mesh.coordinates()]
        self.Ms = Ms
        self.unit_length = unit_length
        #self.rtol = rtol

        domain_classes = {}
        for k in self.domain_ids:
            class DomainK(df.SubDomain):
                def inside(self, pt, on_boundary):
                    return get_domain_id(pt) == k
            domain_classes[k] = DomainK()
        domains = df.CellFunction("size_t", mesh)
        domains.set_all(0)
        for k, d in domain_classes.items():
            d.mark(domains, k)

        self.submeshes = [df.SubMesh(mesh, domains, i) for i in self.domain_ids]
        self.dx = df.Measure("dx")[domains]

        def m_init(pt):
            return m_vals[self.get_domain_id(pt)]

        self.V = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
        self.m = vector_valued_function(m_init, self.V, normalise=True)

    def compute_energies_on_subdomains(self, interaction):
        """
        Given some interaction (such as Zeeman, Demag, Exchange, etc.),
        compute the associated energies on each subdomain as well as the
        total energy.

        *Returns*

        A pair (E_subdmns, E_total), where E_subdmns is a dictionary of
        energies indexed by the subdomain indices, and E_total is the total
        energy of the interaction.

        """
        interaction.setup(self.V, self.m, self.Ms, unit_length=self.unit_length)
        return {k: interaction.compute_energy(dx=self.dx(k)) for k in self.domain_ids},\
            interaction.compute_energy(df.dx)

    def check_energy_consistency(self, interaction):
        E_domains, E_total = self.compute_energies_on_subdomains(interaction)
        finmag.logger.debug("Energies on subdomains: {}".format(E_domains))
        finmag.logger.debug("Sum of energies on subdomains: {}; total energy: {}".format(sum(E_domains.values()), E_total))
        assert np.allclose(sum(E_domains.values()), E_total, atol=0, rtol=1e-12)


def test_energies_in_separated_subdomains(tmpdir):
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

    zeeman = Zeeman(1e6 * np.array([1, 0, 0]))

    mesh = pair_of_disks(d, d, h1, h2, sep, theta=0, maxh=maxh)
    def get_domain_id(pt):
        x, y, z = pt
        return 1 if (np.linalg.norm([x, y]) < 0.5 * (d + sep)) else 2

    m_vals = {1: [1, 0, 0],
              2: [0.5, -0.8, 0]}
    multi_domain_test = MultiDomainTest(mesh, get_domain_id, m_vals, Ms, unit_length=unit_length)
    multi_domain_test.check_energy_consistency(zeeman)


# The same test for a mesh with subdomains that touch will fail for some reason.
# XXX TODO: need to investigate this.
@pytest.mark.xfail
def test_energies_in_touching_subdomains():
    box_mesh = df.BoxMesh(-50, -20, 0, 50, 20, 5, 30, 10, 2)
    def get_domain_id2(pt):
        return 1 if (pt[0] < 0) else 2
    multi_domain_test = MultiDomainTest(box_mesh, get_domain_id, m_vals, Ms, unit_length=unit_length)
    # The next line fails for touching subdomains. Need to investigate this.
    multi_domain_test.check_energy_consistency(zeeman)
