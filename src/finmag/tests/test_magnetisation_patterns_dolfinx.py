"""DOLFINx-parity checks for ``finmag.sim.magnetisation_patterns``.

This slice (SR1 P4-init) restores the module onto the DOLFINx import graph and
validates the FIRST formula family it exposes: the *vortex* family
(``vortex_simple``, ``vortex_feldtkeller`` and their ``initialise_vortex``
dispatcher).  Only the vortex physics is asserted here; the other families
(helix / skyrmion / target) become importable once the shared dolfin-isms are
removed but are intentionally left unvalidated by this slice.

The vortex factories are pure ``math``/tuple callables that ``Simulation.set_m``
interpolates pointwise at the CG1 nodes, so a node's magnetisation equals the
analytic vortex formula evaluated at that node's mesh coordinate.  ``r`` and
``center`` are given in *mesh* coordinates (not metres), matching the legacy
docstring; the mesh below is expressed directly in mesh units.
"""

import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mpi4py import MPI
from dolfinx import mesh

from finmag import Simulation
from finmag.sim.magnetisation_patterns import (
    initialise_vortex,
    vortex_feldtkeller,
    vortex_simple,
)


SRC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SRC_ROOT.parent

MS = 8.6e5
R = 20.0  # vortex core radius, in mesh coordinates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _square_mesh(n=40, half=40.0):
    """A ``[-half, half]^2`` triangle mesh; with ``n=40`` the node pitch is
    ``2.0`` so nodes land exactly on ``(0, 0)``, ``(+/-R, 0)``, ``(0, +/-R)``,
    ``(R/2, 0)`` etc., letting us assert analytic values at exact points."""
    return mesh.create_rectangle(
        MPI.COMM_WORLD, [(-half, -half), (half, half)], [n, n],
        mesh.CellType.triangle,
    )


def _sim(name):
    return Simulation(_square_mesh(), MS, unit_length=1e-9, name=name)


def _coords_values(sim):
    return sim.m_field.coords_and_values()


def _value_at(coords, values, tx, ty, atol=1e-9):
    """Return the nodal magnetisation vector at mesh point ``(tx, ty)``."""
    hit = np.isclose(coords[:, 0], tx, atol=atol) & np.isclose(
        coords[:, 1], ty, atol=atol
    )
    idx = np.nonzero(hit)[0]
    assert idx.size == 1, "expected exactly one node at ({}, {}), got {}".format(
        tx, ty, idx.size
    )
    return values[idx[0]]


# ---------------------------------------------------------------------------
# 1. The module imports cleanly under DOLFINx, pulling in no ``dolfin``.
# ---------------------------------------------------------------------------
def test_module_imports_without_dolfin():
    # Clean subprocess so sys.modules is an exact import-side-effect witness.
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(SRC_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import finmag.sim.magnetisation_patterns as mp\n"
                "assert 'dolfin' not in sys.modules, sorted(sys.modules)\n"
                "assert hasattr(mp, 'vortex_simple')\n"
                "assert hasattr(mp, 'vortex_feldtkeller')\n"
                "assert hasattr(mp, 'initialise_vortex')\n"
            ),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# 2. vortex_simple produces the analytic vortex structure.
# ---------------------------------------------------------------------------
def test_vortex_simple_matches_analytic_profile():
    sim = _sim("vortex_simple")
    sim.set_m(vortex_simple(r=R, center=(0.0, 0.0)))
    coords, values = _coords_values(sim)

    # |m| == 1 everywhere.
    norms = np.linalg.norm(values, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    # Core (rho=0): out-of-plane, polarity up. mz = cos(0) = 1.
    np.testing.assert_allclose(_value_at(coords, values, 0.0, 0.0),
                               [0.0, 0.0, 1.0], atol=1e-12)

    # At rho == r the core has fully rolled over: theta = 2*atan(1) = pi/2,
    # so mz = 0 and the in-plane part is the unit azimuthal (-sin phi, cos phi).
    # (+R, 0): phi=0  -> (0, 1, 0)   (counterclockwise / right-handed curl)
    np.testing.assert_allclose(_value_at(coords, values, R, 0.0),
                               [0.0, 1.0, 0.0], atol=1e-12)
    # (0, +R): phi=pi/2 -> (-1, 0, 0)
    np.testing.assert_allclose(_value_at(coords, values, 0.0, R),
                               [-1.0, 0.0, 0.0], atol=1e-12)
    # (-R, 0): phi=pi -> (0, -1, 0)
    np.testing.assert_allclose(_value_at(coords, values, -R, 0.0),
                               [0.0, -1.0, 0.0], atol=1e-12)

    # Intermediate point (R/2, 0): theta = 2*atan(0.5) => mz=0.6, my=0.8.
    np.testing.assert_allclose(_value_at(coords, values, R / 2.0, 0.0),
                               [0.0, 0.8, 0.6], atol=1e-12)

    # Right-handed in-plane circulation everywhere inside the core:
    # tangential component m . (-sin phi, cos phi) must be >= 0 (curl CCW).
    inside = norms > 0  # all normalised
    rho = np.linalg.norm(coords[:, :2], axis=1)
    core = (rho < R) & (rho > 1e-9)
    phi = np.arctan2(coords[core, 1], coords[core, 0])
    tangential = (values[core, 0] * -np.sin(phi)
                  + values[core, 1] * np.cos(phi))
    assert np.all(tangential > 1e-9)
    # mz strictly positive and decreasing from the core out to r.
    assert np.all(values[core, 2] > 0.0)
    assert inside.all()


# ---------------------------------------------------------------------------
# 3. Chirality and polarity flip the circulation / core direction.
# ---------------------------------------------------------------------------
def test_vortex_simple_chirality_and_polarity_flip():
    # Left-handed (right_handed=False): in-plane part negates, mz unchanged.
    sim = _sim("vortex_lh")
    sim.set_m(vortex_simple(r=R, center=(0.0, 0.0), right_handed=False))
    coords, values = _coords_values(sim)
    # (+R, 0) was (0, 1, 0) right-handed -> (0, -1, 0) left-handed.
    np.testing.assert_allclose(_value_at(coords, values, R, 0.0),
                               [0.0, -1.0, 0.0], atol=1e-12)
    # (0, +R) was (-1, 0, 0) -> (1, 0, 0).
    np.testing.assert_allclose(_value_at(coords, values, 0.0, R),
                               [1.0, 0.0, 0.0], atol=1e-12)
    # Tangential circulation now negative (clockwise).
    rho = np.linalg.norm(coords[:, :2], axis=1)
    core = (rho < R) & (rho > 1e-9)
    phi = np.arctan2(coords[core, 1], coords[core, 0])
    tangential = (values[core, 0] * -np.sin(phi)
                  + values[core, 1] * np.cos(phi))
    assert np.all(tangential < -1e-9)

    # Negative polarity: core flips to point down (mz = -1 at centre), and
    # (polarity<0 and right_handed) also negates the in-plane part.
    sim2 = _sim("vortex_down")
    sim2.set_m(vortex_simple(r=R, center=(0.0, 0.0), polarity=-1))
    coords2, values2 = _coords_values(sim2)
    np.testing.assert_allclose(_value_at(coords2, values2, 0.0, 0.0),
                               [0.0, 0.0, -1.0], atol=1e-12)
    # (R/2, 0): right-handed up was (0, 0.8, 0.6); polarity flip -> (0, -0.8, -0.6).
    np.testing.assert_allclose(_value_at(coords2, values2, R / 2.0, 0.0),
                               [0.0, -0.8, -0.6], atol=1e-12)
    # Core mz strictly negative throughout.
    rho2 = np.linalg.norm(coords2[:, :2], axis=1)
    core2 = (rho2 < R) & (rho2 > 1e-9)
    assert np.all(values2[core2, 2] < 0.0)


# ---------------------------------------------------------------------------
# 4. vortex_feldtkeller: exponential m_z profile.
# ---------------------------------------------------------------------------
def test_vortex_feldtkeller_exponential_profile():
    beta = 20.0
    sim = _sim("vortex_feldtkeller")
    sim.set_m(vortex_feldtkeller(beta=beta, center=(0.0, 0.0)))
    coords, values = _coords_values(sim)

    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-12)

    # Core: mz = exp(0) = 1.
    np.testing.assert_allclose(_value_at(coords, values, 0.0, 0.0),
                               [0.0, 0.0, 1.0], atol=1e-12)

    # (10, 0): mz = exp(-2*100/400) = exp(-0.5); in-plane azimuthal at phi=0.
    mz = math.exp(-0.5)
    mperp = math.sqrt(1.0 - mz * mz)
    np.testing.assert_allclose(_value_at(coords, values, 10.0, 0.0),
                               [0.0, mperp, mz], atol=1e-12)

    # mz falls off monotonically with rho (exponential decay).
    rho = np.linalg.norm(coords[:, :2], axis=1)
    expected_mz = np.exp(-2.0 * rho ** 2 / beta ** 2)
    np.testing.assert_allclose(values[:, 2], expected_mz, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. initialise_vortex dispatcher (exercises the mesh-coordinate accessor:
#    default center == geometric centre of the sample).
# ---------------------------------------------------------------------------
def test_initialise_vortex_default_center_at_sample_centre():
    sim = _sim("init_vortex_simple")
    initialise_vortex(sim, "simple", r=R)  # center defaults to (0, 0)
    coords, values = _coords_values(sim)
    np.testing.assert_allclose(_value_at(coords, values, 0.0, 0.0),
                               [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(_value_at(coords, values, R / 2.0, 0.0),
                               [0.0, 0.8, 0.6], atol=1e-12)

    sim2 = _sim("init_vortex_feldtkeller")
    initialise_vortex(sim2, "feldtkeller", beta=20.0)
    coords2, values2 = _coords_values(sim2)
    np.testing.assert_allclose(_value_at(coords2, values2, 0.0, 0.0),
                               [0.0, 0.0, 1.0], atol=1e-12)

    # A SYMMETRIC mesh cannot distinguish "geometric sample centre" from a
    # hardcoded origin, so pin it on an ASYMMETRIC [0,40]x[0,20] mesh whose
    # geometric centre (20, 10) is NOT the origin: the core (mz=1) must land at
    # (20, 10), and the origin corner must be OUTSIDE the core (mz=0). [Claude Opus 4.8]
    asym = mesh.create_rectangle(
        MPI.COMM_WORLD, [(0.0, 0.0), (40.0, 20.0)], [40, 20],
        mesh.CellType.triangle)
    sim3 = Simulation(asym, MS, unit_length=1e-9, name="init_vortex_asym")
    initialise_vortex(sim3, "simple", r=R)  # center defaults to (20, 10)
    coords3, values3 = _coords_values(sim3)
    np.testing.assert_allclose(_value_at(coords3, values3, 20.0, 10.0),
                               [0.0, 0.0, 1.0], atol=1e-12)
    # origin is a distance 22.36 > R=20 from the core -> outside -> mz == 0
    assert _value_at(coords3, values3, 0.0, 0.0)[2] == pytest.approx(0.0, abs=1e-12)


def test_initialise_vortex_rejects_unknown_type():
    sim = _sim("init_vortex_bad")
    with pytest.raises(ValueError, match="Vortex type must be one of"):
        initialise_vortex(sim, "nonexistent", r=R)
