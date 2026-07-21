"""Focused production tests for the direct DOLFINx restart and output port.

Task 12 slice. Covers:

- restart persistence in the coordinate-aware format (round-trip, cross-instance
  reload, ``t0`` override, and loud mesh-mismatch rejection);
- NDT table output (Tablewriter column/format contract + Tablereader round-trip),
  ported from the M3 ``test_writing_data.py`` invariant;
- VTK/XDMF output routed through the write-only ``Field`` writers, with read-back
  explicitly unavailable by name;
- scheduler integration (``schedule``/``run_until``-driven event loop), and the
  guarantee that ``run_until`` without a schedule does not regress;
- an end-to-end witness with all four ported interactions including FK demag:
  schedule NDT saves, save restart, reload, continue, and assert continuity.
"""

import os

import numpy as np
import pytest
from dolfinx import mesh
from mpi4py import MPI

from finmag.field import Field
from finmag.energies import Exchange, UniaxialAnisotropy, Zeeman
from finmag.sim import sim_helpers
from finmag.sim.sim import Simulation, sim_with
from finmag.util.fileio import Tablereader, Tablewriter


def _box(n=2, length=5.0):
    return mesh.create_box(
        MPI.COMM_WORLD,
        [(0.0, 0.0, 0.0), (length, length, length)],
        [n, n, n],
        mesh.CellType.tetrahedron,
    )


def _make_sim(name="restart_sim", **kwargs):
    kwargs.setdefault("unit_length", 1e-9)
    return Simulation(_box(), 8.6e5, name=name, **kwargs)


# ==========================================================================
# restart
# ==========================================================================

def test_restart_stores_coordinate_aware_format(tmpdir):
    """The restart archive uses the coordinate-aware v2 format (coordinates +
    coordinate-ordered values + mesh hash + parameters), NOT the legacy raw-dof
    layout."""
    sim = _make_sim()
    sim.set_m((0.0, 0.0, 1.0))
    sim.alpha = 0.17
    sim.add(Exchange(13e-12))
    fname = str(tmpdir.join("state.npz"))
    sim.save_restart_data(filename=fname)

    raw = np.load(fname, allow_pickle=True)
    assert set(["coordinates", "m", "mesh_hash", "Ms", "alpha", "gamma",
                "unit_length", "interactions", "simtime", "driver",
                "format_version"]).issubset(set(raw.keys()))
    assert int(raw["format_version"]) == sim_helpers.RESTART_FORMAT_VERSION == 2
    assert str(raw["driver"]) == "scipy"
    # coordinates and values are per-vertex tables (n, gdim) / (n, 3)
    assert raw["coordinates"].ndim == 2 and raw["coordinates"].shape[1] == 3
    assert raw["m"].shape == raw["coordinates"].shape
    assert np.isclose(float(raw["alpha"]), 0.17)
    assert list(raw["interactions"].tolist()) == ["Exchange"]


def test_restart_roundtrip_same_simulation(tmpdir):
    """Save -> mutate -> restart -> state identical (m, t, parameters)."""
    sim = _make_sim()
    sim.set_m(lambda pt: (0.3, 0.4, np.sqrt(1 - 0.25)))
    sim.add(Zeeman((0.0, 0.0, 1e6)))
    sim.run_until(5e-12)

    m_saved = sim.m.copy()
    t_saved = sim.t
    fname = str(tmpdir.join("rt.npz"))
    sim.save_restart_data(filename=fname)

    # mutate both magnetisation and clock
    sim.run_until(20e-12)
    assert not np.allclose(sim.m, m_saved)
    assert sim.t != t_saved

    sim.restart(filename=fname)
    assert np.isclose(sim.t, t_saved)
    assert np.allclose(sim.m, m_saved, atol=1e-12)


def test_restart_t0_override(tmpdir):
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    sim.run_until(3e-12)
    fname = str(tmpdir.join("t0.npz"))
    sim.save_restart_data(filename=fname)

    sim.restart(filename=fname, t0=0.42e-12)
    assert np.isclose(sim.t, 0.42e-12)


def test_restart_cross_instance_same_mesh_recipe(tmpdir):
    """A fresh Simulation built from the same mesh recipe can load the file."""
    sim = _make_sim(name="producer")
    sim.set_m(lambda pt: (np.sin(pt[0]), np.cos(pt[1]), 0.0))
    sim.run_until(2e-12)
    m_saved = sim.m.copy()
    fname = str(tmpdir.join("cross.npz"))
    sim.save_restart_data(filename=fname)

    sim2 = _make_sim(name="consumer")
    sim2.set_m((1.0, 0.0, 0.0))
    sim2.restart(filename=fname)
    assert np.isclose(sim2.t, 2e-12)
    assert np.allclose(sim2.m, m_saved, atol=1e-12)


def test_restart_rejects_mismatched_mesh(tmpdir):
    """Loading into a different mesh is loudly rejected, never misassigned."""
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    fname = str(tmpdir.join("mismatch.npz"))
    sim.save_restart_data(filename=fname)

    other = Simulation(_box(n=3), 8.6e5, unit_length=1e-9, name="other")
    other.set_m((1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="mismatch"):
        other.restart(filename=fname)


def test_load_restart_data_by_simulation_uses_canonical_name(tmpdir):
    os.chdir(str(tmpdir))
    sim = _make_sim(name="canon test")
    sim.set_m((0.0, 1.0, 0.0))
    sim.save_restart_data()  # canonical <sanitized>-restart.npz
    assert os.path.exists("canon_test-restart.npz")
    data = sim_helpers.load_restart_data(sim)
    assert str(data["simname"]) == "canon test"
    assert data["m"].shape[1] == 3


# ==========================================================================
# NDT output
# ==========================================================================

def test_ndt_format_and_roundtrip(tmpdir):
    """Tablewriter writes the legacy .ndt column/format contract and Tablereader
    reads exactly what was written."""
    sim = _make_sim()
    sim.set_m((0.0, 0.0, 1.0))
    ndt = str(tmpdir.join("data.ndt"))
    writer = Tablewriter(ndt, sim, override=True)
    writer.save()

    with open(ndt) as handle:
        header, units = handle.readline(), handle.readline()
    # comment symbol, canonical column names and units row
    assert header.startswith("# ")
    for col in ("time", "m_x", "m_y", "m_z"):
        assert col in header.split()
    assert "<s>" in units.split() and "<>" in units.split()

    reader = Tablereader(ndt)
    assert np.isclose(reader["time"][0], sim.t)
    m = np.array(reader["m_x", "m_y", "m_z"])[:, 0]
    assert np.allclose(m, sim.m_average, atol=1e-9)


def test_ndt_float_format_precision(tmpdir):
    """The float format is the legacy 12-significant-figure ``%18.12g``."""
    sim = _make_sim()
    ndt = str(tmpdir.join("fmt.ndt"))
    writer = Tablewriter(ndt, sim, override=True)
    assert writer.float_format == "%18.12g "
    assert writer.string_format == "%18s "
    assert writer.comment_symbol == "# "
    assert (writer.float_format % (1.0 / 3.0)).strip() == "0.333333333333"


def test_sim_save_averages_appends_rows(tmpdir):
    sim = _make_sim(name="ndt_sim")
    sim.set_m((1.0, 0.0, 0.0))
    sim.ndtfilename = str(tmpdir.join("ndt_sim.ndt"))
    sim.save_averages()
    sim.add(Zeeman((0.0, 0.0, 1e6)))
    sim.run_until(2e-12)
    sim.save_ndt()  # alias of save_averages

    reader = Tablereader(sim.ndtfilename)
    assert len(reader["time"]) == 2
    assert reader["time"][1] > reader["time"][0]


# ==========================================================================
# VTK / XDMF output
# ==========================================================================

def test_save_vtk_writes_pvd(tmpdir):
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    pvd = str(tmpdir.join("m.pvd"))
    sim.save_vtk(filename=pvd)
    sim.m_field.close_pvd()
    assert os.path.exists(pvd)


def test_save_field_to_vtk_xdmf(tmpdir):
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    xdmf = str(tmpdir.join("m.xdmf"))
    sim.save_field_to_vtk("m", filename=xdmf)
    sim.m_field.close_xdmf()
    assert os.path.exists(xdmf)


def test_vtk_readback_unavailable_by_name():
    """VTK/XDMF is write-only; legacy dolfinh5tools read-back fails by name."""
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    with pytest.raises(NotImplementedError, match="HDF5|dolfinh5tools"):
        sim.m_field.save_hdf5("x.h5")


# ==========================================================================
# scheduler
# ==========================================================================

def test_run_until_without_schedule_no_regression():
    """With no schedule, run_until still advances physical time exactly."""
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    sim.add(Zeeman((0.0, 0.0, 1e6)))
    sim.run_until(3e-12)
    assert np.isclose(sim.t, 3e-12)


def test_schedule_save_ndt_every(tmpdir):
    sim = _make_sim(name="sched")
    sim.set_m((1.0, 0.0, 0.0))
    sim.add(Zeeman((0.0, 0.0, 1e6)))
    sim.ndtfilename = str(tmpdir.join("sched.ndt"))
    sim.schedule("save_ndt", every=2e-12)
    sim.run_until(1e-11)

    reader = Tablereader(sim.ndtfilename)
    times = reader["time"]
    # saves at 0, 2e-12, ..., 1e-11 -> 6 rows
    assert len(times) == 6
    assert np.isclose(times[0], 0.0)
    assert np.isclose(times[-1], 1e-11)
    assert np.all(np.diff(times) > 0)


def test_schedule_callable_and_clear(tmpdir):
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    sim.add(Zeeman((0.0, 0.0, 1e6)))
    hits = []
    sim.schedule(lambda s: hits.append(s.t), every=5e-12)
    sim.run_until(1e-11)
    assert len(hits) == 3  # t = 0, 5e-12, 1e-11
    sim.clear_schedule()
    sim.run_until(2e-11)
    assert len(hits) == 3  # cleared: no further callbacks


def test_schedule_unknown_shortcut_raises_by_name():
    sim = _make_sim()
    with pytest.raises(KeyError, match="unknown"):
        sim.schedule("definitely_not_a_shortcut", every=1e-12)


def test_unschedule_removes_item(tmpdir):
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    sim.add(Zeeman((0.0, 0.0, 1e6)))
    hits = []
    item = sim.schedule(lambda s: hits.append(s.t), every=5e-12)
    sim.unschedule(item)
    sim.run_until(1e-11)
    assert hits == []


# ==========================================================================
# end-to-end witness: all four interactions incl FK demag
# ==========================================================================

@pytest.mark.filterwarnings("ignore")
def test_end_to_end_all_interactions_with_restart(tmpdir):
    """All four ported interactions (Exchange, Zeeman, UniaxialAnisotropy, FK
    demag): schedule NDT saves, save restart, reload into a fresh simulation,
    continue, and assert continuity of t and m across the reload."""
    os.chdir(str(tmpdir))
    sim = sim_with(
        _box(n=2), Ms=8.6e5, m_init=(1.0, 0.0, 0.0), alpha=0.5,
        unit_length=1e-9, A=13e-12, K1=1e5, K1_axis=(0.0, 0.0, 1.0),
        H_ext=(0.0, 0.0, 1e6), demag_solver="FK", name="e2e",
    )
    assert sim.has_interaction("Demag")
    sim.ndtfilename = "e2e.ndt"
    sim.schedule("save_ndt", every=2e-12)
    sim.run_until(6e-12)

    reader = Tablereader("e2e.ndt")
    assert len(reader["time"]) >= 3

    m_saved = sim.m.copy()
    t_saved = sim.t
    sim.save_restart_data(filename="e2e.npz")

    # fresh simulation, same recipe: reload and continue
    sim2 = sim_with(
        _box(n=2), Ms=8.6e5, m_init=(0.0, 1.0, 0.0), alpha=0.5,
        unit_length=1e-9, A=13e-12, K1=1e5, K1_axis=(0.0, 0.0, 1.0),
        H_ext=(0.0, 0.0, 1e6), demag_solver="FK", name="e2e2",
    )
    sim2.restart(filename="e2e.npz")
    assert np.isclose(sim2.t, t_saved)
    assert np.allclose(sim2.m, m_saved, atol=1e-12)

    sim2.run_until(1e-11)
    assert sim2.t >= 1e-11
    # unit norm preserved through the reload+continue
    m_nodal = sim2.m.reshape((3, -1))
    assert np.allclose(np.linalg.norm(m_nodal, axis=0), 1.0, atol=1e-4)
