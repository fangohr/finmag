"""Witness that the dolfin-free Magpar readers load the checked-in reference
data correctly under the DOLFINx environment (SR1 P5.2 precondition).

These tests exercise ``finmag.util.magpar_io`` -- the extracted, dolfin-free
readers -- against the three checked-in Magpar reference datasets
(``test_exch``, ``test_anis``, ``test_demag``). Importing the module at all is
itself part of the contract: the legacy ``finmag.util.magpar`` cannot be
imported without ``dolfin``, so a plain ``import`` here proves the readers are
usable in the supported environment. [Claude Opus 4.8]
"""

import os

import numpy as np
import pytest

# The bare import is a contract: it must succeed with no legacy ``dolfin``.
from finmag.util import magpar_io

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# (relative base path, magpar field key, expected node count) for each dataset.
DATASETS = [
    ("exchange/magpar_result/test_exch", "exch", 369),
    ("anisotropy/magpar_result/test_anis", "anis", 1537),
    ("demag/magpar_result/test_demag", "demag", 1084),
]


def _base(rel):
    return os.path.join(MODULE_DIR, rel)


@pytest.mark.parametrize("rel,field_key,n_nodes", DATASETS)
def test_read_femsh_node_table(rel, field_key, n_nodes):
    nodes, connectivity = magpar_io.read_femsh(_base(rel) + ".0001.femsh")
    assert nodes.shape == (n_nodes, 3)
    assert connectivity.shape[1] == 4
    assert np.isfinite(nodes).all()


@pytest.mark.parametrize("rel,field_key,n_nodes", DATASETS)
def test_get_field_shape_and_coords(rel, field_key, n_nodes):
    nodes, field = magpar_io.get_field(_base(rel), field_key)
    # get_field returns coordinates (n, 3) and a flat component-blocked field 3n.
    assert nodes.shape == (n_nodes, 3)
    assert field.shape == (3 * n_nodes,)
    assert np.isfinite(field).all()


def test_get_field_applies_mu0_conversion():
    """get_field must divide Magpar's Tesla values by mu0 to give A/m."""
    base = _base("exchange/magpar_result/test_exch")
    raw = magpar_io.read_inp(base + ".0001")  # Tesla, per-column
    _, field = magpar_io.get_field(base, "exch")  # A/m, component-blocked
    n = raw["Hexch_x"].shape[0]
    mu0 = np.pi * 4e-7
    # Component-blocked layout: [Hx.., Hy.., Hz..]; first block is Hexch_x/mu0.
    np.testing.assert_allclose(field[:n], raw["Hexch_x"] / mu0, rtol=1e-12)
    np.testing.assert_allclose(field[n:2 * n], raw["Hexch_y"] / mu0, rtol=1e-12)
    np.testing.assert_allclose(field[2 * n:], raw["Hexch_z"] / mu0, rtol=1e-12)


def test_unknown_field_raises():
    base = _base("exchange/magpar_result/test_exch")
    with pytest.raises(NotImplementedError):
        magpar_io.get_field(base, "not_a_field")
