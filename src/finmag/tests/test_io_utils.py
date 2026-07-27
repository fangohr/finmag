"""Task 26a: decision-free I/O/utility parity wins.

Formerly ``src/finmag/tests/test_io_utils_dolfinx.py``; renamed onto the
sibling-free canonical name (canonical-test-paths move, 2026-07-27).
No master ancestor is removed by this rename. ``field_test.py``'s two
``test_probe_*`` functions are covered by the ported ``src/finmag/field_test.py``
(that file's own move), and ``tests/test_skyrmions.py`` is RETAINED: the header
below records it as a placeholder stub with no real coverage to map, but no
named port function covers ``test_skyrmion``, so the conservative rule keeps
the file in the tree (it fails in the non-gating inventory lane by design).

Covers the three legacy capabilities restored in this slice, each either a
faithful mechanical port (point evaluation) or a pure analytic transcription
(spherical conversion, skyrmion number) -- none required an owner decision.

- ``Field.probe``/``Field.__call__``: point-in-cell evaluation, promoted from
  the Task 30 ``examples/exchange_demag/test_exchange_demag.py``
  ``_eval_scalar_function`` workaround into the shared
  ``finmag.field.evaluate_at_point`` helper (also exercised directly here on a
  raw ``dolfinx.fem.Function``, matching how
  ``EnergyBase.energy_density_function()`` returns one "to allow probing",
  exactly as legacy did).
- ``Field.get_spherical``: transcribed verbatim from legacy's
  ``theta = atan2(m_r, m_z)`` / ``phi = atan2(m_y, m_x)`` nodal formulas.
- ``Simulation.skyrmion_number``/``skyrmion_number_density_function``:
  transcribed verbatim from legacy's
  ``-1/(4*pi) integral(m . (dm/dx x dm/dy))`` formula (2D mesh: whole domain;
  3D mesh: top surface only), restored after being dropped outright in
  Task 9.

MASTER -> PORT MAPPING-HEADER (BUCKET-B: mixed ancestor/no-ancestor file).

1) ``Field.probe`` / ``Field.__call__`` part -- HAS a master ancestor, but
   that ancestor is fully covered elsewhere, not here:

   - ``field_test.py::TestField.test_probe_scalar_field`` (mesh_dim 1/2/3,
     at-node + off-node ``self.probing_coord = 0.4351``, ``self.tol1 =
     5e-13``)
       -> already transcribed VERBATIM, tolerance UNCHANGED (``tol1 =
          5e-13``), as
          ``field_test.py::TestField.test_probe_scalar_field``
          (line ~959).
   - ``field_test.py::TestField.test_probe_vector_field`` (``vector3d_fspaces``,
     same at-node/off-node pattern, ``tol1``)
       -> already transcribed VERBATIM, tolerance UNCHANGED, as
          ``field_test.py::TestField.test_probe_vector_field``
          (line ~988).

   These are the ONLY two ``test_probe_*``-named functions in master's
   ``field_test.py::TestField`` (checked via ``git show b5015c5a`` --
   grep for ``def test_``). Both are already ported with real assertions
   and unchanged tolerance in ``field_test.py``, which per its own
   docstring transcribes the full ``TestField`` suite. Nothing is dropped,
   so nothing is restored here -- duplicating them in this file would just
   be redundant coverage of the same two master functions.

   The probe tests that DO live in *this* file are NOT master transcriptions:
   they are NEW tests (Task 26a/30 provenance) exercising the promoted
   ``evaluate_at_point`` helper along axes master's ``field_test.py`` never
   tested standalone -- ``RuntimeError`` message on an out-of-mesh point,
   ``__call__`` as shorthand for ``.probe``, a raw (non-``Field``-wrapped)
   ``dolfinx.fem.Function`` (mirroring what
   ``EnergyBase.energy_density_function()`` returns), and one-point-per-call
   sampling along a line (the exchange_demag regression witness):
       test_probe_vector_field_matches_analytic_value
       test_probe_scalar_field_returns_python_float
       test_call_is_shorthand_for_probe
       test_probe_outside_mesh_raises_runtime_error
       test_evaluate_at_point_works_on_a_raw_function_not_just_field
       test_probe_along_the_exchange_demag_sampling_line

2) ``Field.get_spherical`` / ``Simulation.skyrmion_number`` part -- NO real
   master ancestor test exists for either:

   - ``Field.get_spherical`` itself exists in master
     (``git show b5015c5a:src/finmag/field.py``, ~line 680, the
     ``theta = arctan(m_r / m_z)`` / ``phi = arctan(m_y / m_x)`` docstring
     this port's formulas are transcribed from), but master's
     ``field_test.py`` (grepped for ``get_spherical``) has ZERO tests that
     exercise it -- no ancestor test to map.
   - ``Simulation.skyrmion_number`` / ``skyrmion_number_density_function``
     are real master methods (``sim_helpers.skyrmion_number`` /
     ``skyrmion_number_density_function``, wired onto ``Simulation`` in
     ``sim/sim.py``), but master's dedicated test module,
     ``git show b5015c5a:src/finmag/tests/test_skyrmions.py``, is CONFIRMED
     a placeholder stub: its only function, ``test_skyrmion()``, asserts
     ``abs(1e-5 - 1e-5) < 1e-6`` -- literal self-consistent dummy numbers,
     no simulation, no skyrmion, no real ancestor coverage to map or
     restore.

   With no master test to transcribe, this port validates both against
   independent ANALYTIC references instead (see the
   ``# ===== NEW under DOLFINx =====`` banner below):
     - ``get_spherical``: five hand-picked unit vectors along/off the axes
       with closed-form ``(theta, phi)`` (e.g. ``+z -> (0, 0)``,
       ``+x -> (pi/2, 0)``, ``(1,1,0)/sqrt(2) -> (pi/2, pi/4)``), plus a
       round-trip reconstruction
       (``sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)`` recovers the
       original unit vector) for a non-axis-aligned direction, plus a
       scalar-field rejection check.
     - ``skyrmion_number``: the analytic topological-charge invariant that a
       compactly-supported Bloch skyrmion ansatz (radius 20, reproduced
       inline from legacy's untouched, dolfin-only
       ``sim.magnetisation_patterns.initialise_skyrmions`` profile, since
       that module cannot be imported under DOLFINx) carries topological
       charge magnitude -> 1 as the mesh is refined (measured: n=20 ->
       0.8668, n=40 -> 0.9651, n=80 -> 0.9912, monotonic convergence),
       is exactly 0 for a uniform state (2D and 3D top-surface paths),
       the 3D top-surface integration path numerically matches the 2D
       whole-domain path for a z-invariant profile with the same xy
       footprint (measured: both give 0.86675204942... at n=20), and the
       lumped nodal density field integrates (nodal-volume-weighted sum)
       back to the whole-mesh scalar exactly.
"""

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from finmag import Simulation
from finmag.field import Field, associated_scalar_space, evaluate_at_point
from finmag.energies.energy_base import _nodal_volume_owned


# --------------------------------------------------------------------------
# Field.probe / Field.__call__ (point-in-cell evaluation)
# --------------------------------------------------------------------------

@pytest.fixture
def box():
    return mesh.create_box(
        MPI.COMM_WORLD,
        [(0.0, 0.0, 0.0), (10.0, 10.0, 10.0)],
        [4, 4, 4],
        mesh.CellType.tetrahedron,
    )


def test_probe_vector_field_matches_analytic_value(box):
    S3 = fem.functionspace(box, ("Lagrange", 1, (3,)))
    field = Field(S3, lambda x: np.vstack([x[0], 2.0 * x[1], 3.0 * x[2]]))
    value = field.probe((3.0, 4.0, 5.0))
    assert isinstance(value, np.ndarray)
    assert value.shape == (3,)
    np.testing.assert_allclose(value, [3.0, 8.0, 15.0], atol=1e-10)


def test_probe_scalar_field_returns_python_float(box):
    S1 = fem.functionspace(box, ("Lagrange", 1))
    field = Field(S1, lambda x: x[0] + x[1])
    value = field.probe((3.0, 4.0, 5.0))
    assert isinstance(value, float)
    assert abs(value - 7.0) < 1e-10


def test_call_is_shorthand_for_probe(box):
    S1 = fem.functionspace(box, ("Lagrange", 1))
    field = Field(S1, lambda x: x[0] + x[1])
    assert field((3.0, 4.0, 5.0)) == field.probe((3.0, 4.0, 5.0))


def test_probe_outside_mesh_raises_runtime_error(box):
    S1 = fem.functionspace(box, ("Lagrange", 1))
    field = Field(S1, 1.0)
    with pytest.raises(RuntimeError, match="not inside the mesh"):
        field.probe((100.0, 100.0, 100.0))
    with pytest.raises(RuntimeError, match="not inside the mesh"):
        field((100.0, 100.0, 100.0))


def test_evaluate_at_point_works_on_a_raw_function_not_just_field(box):
    """``energy_density_function()`` returns a raw ``fem.Function`` (legacy
    also returned a raw ``dolfin.Function`` "to allow probing"). The shared
    helper promoted from the exchange_demag workaround must work directly on
    such a raw Function, not only through ``Field``."""
    S1 = fem.functionspace(box, ("Lagrange", 1))
    raw_function = fem.Function(S1)
    # Linear (not quadratic) so CG1 point-in-cell evaluation is exact, not
    # merely nodally-exact-but-interpolated.
    raw_function.interpolate(lambda x: 2.0 * x[0] + x[1])
    value = evaluate_at_point(raw_function, (3.0, 4.0, 5.0))
    assert isinstance(value, float)
    assert abs(value - 10.0) < 1e-10


def test_probe_along_the_exchange_demag_sampling_line(box):
    """Regression witness for the reverted exchange_demag example: sample a
    scalar Function at many points along a line, one point per call, exactly
    as the restored example (and legacy) do."""
    S1 = fem.functionspace(box, ("Lagrange", 1))
    field = Field(S1, lambda x: x[2])
    line = [(5.0, 5.0, float(z)) for z in range(11)]
    sampled = [field.probe(p) for p in line]
    np.testing.assert_allclose(sampled, list(range(11)), atol=1e-10)


# ===== NEW under DOLFINx =====
# No master ancestor test exists for either section below: master's
# ``field_test.py`` has zero tests exercising ``get_spherical`` (grepped), and
# master's ``tests/test_skyrmions.py`` is a confirmed placeholder stub (see
# module docstring). Both sections validate against independent analytic
# references instead of transcribing a master test.

# --------------------------------------------------------------------------
# Field.get_spherical
# --------------------------------------------------------------------------

@pytest.fixture
def cube():
    return mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)


def _uniform_vector_field(functionspace, vector):
    vector = np.asarray(vector, dtype=np.float64)
    return Field(functionspace, lambda x: np.repeat(vector[:, None], x.shape[1], axis=1))


@pytest.mark.parametrize(
    "vector, expected_theta, expected_phi",
    [
        ((0.0, 0.0, 1.0), 0.0, 0.0),
        ((1.0, 0.0, 0.0), np.pi / 2, 0.0),
        ((0.0, 1.0, 0.0), np.pi / 2, np.pi / 2),
        ((0.0, 0.0, -1.0), np.pi, 0.0),
        ((1.0, 1.0, 0.0), np.pi / 2, np.pi / 4),
    ],
)
def test_get_spherical_analytic_points(cube, vector, expected_theta, expected_phi):
    S3 = fem.functionspace(cube, ("Lagrange", 1, (3,)))
    field = _uniform_vector_field(S3, vector)
    theta, phi = field.get_spherical()

    assert isinstance(theta, fem.Function)
    assert isinstance(phi, fem.Function)
    assert theta is field.theta
    assert phi is field.phi

    np.testing.assert_allclose(theta.x.array, expected_theta, atol=1e-10)
    np.testing.assert_allclose(phi.x.array, expected_phi, atol=1e-10)


def test_get_spherical_round_trip_reconstructs_unit_vector(cube):
    S3 = fem.functionspace(cube, ("Lagrange", 1, (3,)))
    vector = np.array([0.3, -0.6, 0.74327])
    vector = vector / np.linalg.norm(vector)
    field = _uniform_vector_field(S3, vector)
    theta, phi = field.get_spherical()

    reconstructed = np.array(
        [
            np.sin(theta.x.array) * np.cos(phi.x.array),
            np.sin(theta.x.array) * np.sin(phi.x.array),
            np.cos(theta.x.array),
        ]
    ).T
    np.testing.assert_allclose(reconstructed, np.tile(vector, (reconstructed.shape[0], 1)), atol=1e-10)


def test_get_spherical_rejects_scalar_field(cube):
    S1 = fem.functionspace(cube, ("Lagrange", 1))
    field = Field(S1, 1.0)
    with pytest.raises(ValueError, match="3-component"):
        field.get_spherical()


# --------------------------------------------------------------------------
# Simulation.skyrmion_number / skyrmion_number_density_function
# --------------------------------------------------------------------------

SKYRMION_RADIUS = 20.0


def _bloch_skyrmion(pos):
    """Compactly-supported Bloch-type skyrmion ansatz, radius SKYRMION_RADIUS,
    core pointing down (``m_z=-1``) at the centre, ``m_z=+1`` background.
    Faithful transcription of the (untouched, legacy-dolfin-only, off the
    DOLFINx import graph) ``finmag.sim.magnetisation_patterns.
    initialise_skyrmions`` profile, reproduced inline here as self-contained
    analytic test data (that module cannot be imported under DOLFINx -- it
    still does ``import dolfin``)."""
    x, y = pos[0], pos[1]
    r = np.sqrt(x * x + y * y)
    clipped_r = np.clip(r, 0.0, SKYRMION_RADIUS)
    theta = np.arctan2(y, x)
    mz = np.where(r > SKYRMION_RADIUS, 1.0, -np.cos(np.pi * clipped_r / SKYRMION_RADIUS))
    m_perp = np.where(r > SKYRMION_RADIUS, 0.0, np.sin(np.pi * clipped_r / SKYRMION_RADIUS))
    mx = -np.sin(theta) * m_perp
    my = np.cos(theta) * m_perp
    out = np.vstack([mx, my, mz])
    norm = np.linalg.norm(out, axis=0)
    norm = np.where(norm == 0.0, 1.0, norm)
    return out / norm


def _disk_footprint_rectangle_mesh(n):
    return mesh.create_rectangle(
        MPI.COMM_WORLD, [(-40.0, -40.0), (40.0, 40.0)], [n, n], mesh.CellType.triangle
    )


def test_skyrmion_number_is_zero_for_uniform_state_2d():
    domain = _disk_footprint_rectangle_mesh(10)
    sim = Simulation(domain, 8.6e5, unit_length=1e-9, name="sk_uniform_2d")
    sim.set_m((0.0, 0.0, 1.0))
    assert abs(sim.skyrmion_number()) < 1e-10


def test_skyrmion_number_is_zero_for_uniform_state_3d_top_surface_path():
    domain = mesh.create_box(
        MPI.COMM_WORLD, [(-40.0, -40.0, 0.0), (40.0, 40.0, 5.0)], [10, 10, 2],
        mesh.CellType.tetrahedron,
    )
    sim = Simulation(domain, 8.6e5, unit_length=1e-9, name="sk_uniform_3d")
    sim.set_m((0.0, 0.0, 1.0))
    assert abs(sim.skyrmion_number()) < 1e-10


def test_skyrmion_number_approaches_unit_topological_charge_2d():
    """A resolved Bloch skyrmion gives a topological charge that converges to
    1 in magnitude as the mesh is refined (measured: n=20 -> 0.8668,
    n=40 -> 0.9651, n=80 -> 0.9912 -- monotonic convergence towards 1)."""
    coarse = Simulation(_disk_footprint_rectangle_mesh(20), 8.6e5, unit_length=1e-9,
                         name="sk_coarse")
    coarse.set_m(_bloch_skyrmion)
    sk_coarse = coarse.skyrmion_number()

    fine = Simulation(_disk_footprint_rectangle_mesh(40), 8.6e5, unit_length=1e-9,
                       name="sk_fine")
    fine.set_m(_bloch_skyrmion)
    sk_fine = fine.skyrmion_number()

    assert 0.75 < sk_coarse < 1.05
    assert 0.9 < sk_fine < 1.05
    # Refinement must move the discrete value closer to the analytic +/-1.
    assert abs(sk_fine - 1.0) < abs(sk_coarse - 1.0)


def test_skyrmion_number_3d_top_surface_matches_2d_projection():
    """The 3D top-surface integration path must reduce to the same value as
    the 2D whole-domain path for a z-invariant profile with a matching xy
    footprint/resolution (measured: both give 0.8667520494268... at n=20)."""
    two_d = Simulation(_disk_footprint_rectangle_mesh(20), 8.6e5, unit_length=1e-9,
                        name="sk_2d_ref")
    two_d.set_m(_bloch_skyrmion)
    sk_2d = two_d.skyrmion_number()

    three_d = Simulation(
        mesh.create_box(
            MPI.COMM_WORLD, [(-40.0, -40.0, 0.0), (40.0, 40.0, 5.0)], [20, 20, 2],
            mesh.CellType.tetrahedron,
        ),
        8.6e5, unit_length=1e-9, name="sk_3d_top",
    )
    three_d.set_m(_bloch_skyrmion)
    sk_3d = three_d.skyrmion_number()

    assert abs(sk_3d - sk_2d) < 1e-8


def test_skyrmion_number_density_function_integrates_to_skyrmion_number():
    """The lumped nodal density, weighted by nodal volume and summed, must
    reproduce the whole-mesh skyrmion number exactly (both are derived from
    the same box-assembled lumped machinery)."""
    sim = Simulation(_disk_footprint_rectangle_mesh(20), 8.6e5, unit_length=1e-9,
                      name="sk_density")
    sim.set_m(_bloch_skyrmion)

    density_function = sim.skyrmion_number_density_function()
    assert isinstance(density_function, fem.Function)

    S1 = associated_scalar_space(sim.S3)
    nodal_volume = _nodal_volume_owned(S1)
    owned_density = density_function.x.array[: nodal_volume.size]
    integrated = float(np.sum(owned_density * nodal_volume))

    assert abs(integrated - sim.skyrmion_number()) < 1e-9
