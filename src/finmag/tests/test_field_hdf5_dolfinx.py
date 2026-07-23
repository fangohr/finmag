"""HDF5 checkpoint round-trip for the DOLFINx-backed :class:`Field`.

These tests pin the single-file, self-describing, coordinate-aware ``.h5``
round-trip added for SR1 P4-hdf5. The on-disk layout mirrors the restart v2
coordinate-aware format (owned vertex coordinates + coordinate-ordered values),
so a load remaps stored values onto the target space by physical coordinate and
is therefore robust to FEM/vertex reordering -- never a raw dof-index copy.
"""

import os

import h5py
import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from finmag.field import Field


def _scalar_space(n=3):
    domain = mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    return fem.functionspace(domain, ("Lagrange", 1))


def _vector_space(n=3):
    domain = mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    return fem.functionspace(domain, ("Lagrange", 1, (3,)))


def _node_varying_vector(space):
    # Spatially varying so an ordering/scramble bug cannot hide behind a
    # constant field.
    return Field(
        space,
        lambda x: np.vstack((1.0 + x[0], 2.0 + x[1], 3.0 + x[0] * x[1])),
        name="m",
    )


def test_hdf5_roundtrip_node_varying_vector_exact(tmpdir):
    """A node-varying vector Field round-trips bit-for-bit on the same space."""
    space = _vector_space()
    field = _node_varying_vector(space)
    path = str(tmpdir.join("m.h5"))

    field.save_hdf5(path)
    reloaded = Field.from_hdf5(space, path)

    assert np.array_equal(
        reloaded.get_ordered_numpy_array_xxx(),
        field.get_ordered_numpy_array_xxx(),
    )


def test_hdf5_roundtrip_scalar_exact(tmpdir):
    """A node-varying scalar Field round-trips bit-for-bit."""
    space = _scalar_space()
    field = Field(space, lambda x: 0.5 + x[0] - 2.0 * x[1], name="phi")
    path = str(tmpdir.join("s.h5"))

    field.save_hdf5(path)
    reloaded = Field.from_hdf5(space, path)

    assert reloaded.value_dim() == 1
    assert np.array_equal(
        reloaded.get_ordered_numpy_array_xyz(),
        field.get_ordered_numpy_array_xyz(),
    )


def test_hdf5_load_is_coordinate_aware_not_row_order(tmpdir):
    """Load must remap by coordinate, not by stored row position.

    Mirrors ``test_restart_remap_undoes_nonidentity_coordinate_permutation``:
    apply the SAME permutation to the coordinate and value rows (preserving each
    (coordinate, value) pairing) so only the file row order changes. A correct
    coordinate-aware load still restores the exact original nodal values; a
    row-position load would scramble them.
    """
    space = _vector_space()
    field = _node_varying_vector(space)
    original = field.get_ordered_numpy_array_xxx().copy()
    path = str(tmpdir.join("m.h5"))
    field.save_hdf5(path)

    with h5py.File(path, "r+") as h5:
        coords = np.asarray(h5["coordinates"])
        values = np.asarray(h5["values"])
        n = coords.shape[0]
        rng = np.random.RandomState(20260724)  # fixed seed: reproducible
        perm = rng.permutation(n)
        assert not np.array_equal(perm, np.arange(n)), (
            "shuffle must be non-identity for this test to pin anything")
        del h5["coordinates"]
        del h5["values"]
        h5.create_dataset("coordinates", data=coords[perm])
        h5.create_dataset("values", data=values[perm])

    reloaded = Field.from_hdf5(space, path)
    assert np.allclose(
        reloaded.get_ordered_numpy_array_xxx(), original, atol=1e-12)

    # And the reloaded values land on the correct physical nodes.
    coords_out, vals_out = reloaded.coords_and_values()
    expected = np.column_stack((
        1.0 + coords_out[:, 0],
        2.0 + coords_out[:, 1],
        3.0 + coords_out[:, 0] * coords_out[:, 1],
    ))
    assert np.allclose(vals_out, expected)


def test_hdf5_is_single_self_describing_file(tmpdir):
    """The ``.h5`` is one valid, self-describing HDF5 file: no JSON sidecar."""
    space = _vector_space()
    field = _node_varying_vector(space)
    path = str(tmpdir.join("m.h5"))

    field.save_hdf5(path, t=1.5, unit_length=1e-9)

    assert os.path.exists(path)
    # No metadata sidecar of any spelling is required to read it back.
    assert not os.path.exists(path.replace(".h5", ".json"))
    assert not os.path.exists(path + ".json")

    with h5py.File(path, "r") as h5:
        assert h5.attrs["format"] == "finmag-field-hdf5"
        assert int(h5.attrs["format_version"]) == 1
        assert int(h5.attrs["value_dim"]) == 3
        assert float(h5.attrs["t"]) == 1.5
        assert float(h5.attrs["unit_length"]) == 1e-9
        assert "coordinates" in h5
        assert "values" in h5
        n = h5["coordinates"].shape[0]
        assert h5["values"].shape == (n, 3)

    # Reading back needs only this one file.
    reloaded = Field.from_hdf5(space, path)
    assert reloaded.allclose(field)


def test_hdf5_appends_extension(tmpdir):
    """``save_hdf5``/``from_hdf5`` normalise a missing ``.h5`` suffix."""
    space = _vector_space()
    field = _node_varying_vector(space)
    base = str(tmpdir.join("noext"))

    field.save_hdf5(base)
    assert os.path.exists(base + ".h5")

    reloaded = Field.from_hdf5(space, base)
    assert reloaded.allclose(field)


def test_hdf5_witness_and_xdmf_still_works(tmpdir):
    """h5py round-trip works AND the pre-existing XDMF writer is unchanged."""
    import h5py as _h5py  # witness: the dependency is importable

    assert _h5py is not None
    space = _vector_space()
    field = _node_varying_vector(space)

    hpath = str(tmpdir.join("w.h5"))
    field.save_hdf5(hpath)
    assert Field.from_hdf5(space, hpath).allclose(field)

    # No regression to the existing XDMF/HDF5-backed writer.
    xpath = str(tmpdir.join("w.xdmf"))
    field.save_xdmf(xpath)
    field.close_xdmf()
    assert os.path.exists(xpath)


def test_hdf5_rejects_mismatched_value_dim(tmpdir):
    """Loading a vector checkpoint into a scalar space is a loud error."""
    vec = _vector_space()
    field = _node_varying_vector(vec)
    path = str(tmpdir.join("m.h5"))
    field.save_hdf5(path)

    scalar = _scalar_space()
    with pytest.raises(ValueError, match="value_dim|component"):
        Field.from_hdf5(scalar, path)
