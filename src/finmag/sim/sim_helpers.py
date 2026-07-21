"""Simulation helper functions, ported directly to DOLFINx.

This module holds the restart persistence helpers (``save_restart_data`` /
``load_restart_data``) plus a handful of pure-Python simulation utilities. It is
import-clean in the DOLFINx environment: legacy ``dolfin``, ``finmag.native``
and ``finmag.util.meshes`` are not imported at all -- the DOLFIN-specific
helpers that used to live here (``skyrmion_number`` etc.) were removed as part
of this port and are not part of the ported core ``Simulation`` surface.

Restart format decision (deliberate deviation from the legacy npz layout,
documented in ``transition-notes.org`` and ``dev/dolfinx/porting_map.md``):

The legacy restart file stored the magnetisation as a raw, backend-dof-ordered
owned array (``_m_field.get_numpy_array_debug()``). That layout is only valid
for the *identical* in-memory function-space dof numbering, so restarting into a
differently-ordered rebuild of the same mesh could silently misassign values.
The DOLFINx port instead stores the magnetisation in a stable, coordinate-aware
format: the owned vertex coordinates alongside the coordinate-ordered vector
values, plus a mesh hash and the material parameters and interaction list. On
load the values are remapped to the target field by coordinate, so a rebuild of
the same mesh with a different vertex/dof numbering is either correctly remapped
or loudly rejected -- never silently misassigned. [Claude Opus 4.8]
"""

import logging
import os
import re
import shutil
import hashlib
from datetime import datetime

import numpy as np

import finmag

log = logging.getLogger("finmag")

# Version tag written into every DOLFINx-era restart archive. v1 was the legacy
# raw-dof npz layout; v2 is the coordinate-aware format defined here.
RESTART_FORMAT_VERSION = 2

# Coordinate matching tolerance (mesh units) used when remapping a restart file
# onto a target mesh. Coordinates are rounded to this many decimals for hashing
# and lookup, matching the Field coordinate-ordering convention.
_COORD_DECIMALS = 12


def clean_filename(filename):
    """Remove non-alphanumeric characters from a filename.

    Reimplemented locally (identical to ``finmag.util.helpers.clean_filename``)
    so the restart helpers do not import the legacy ``dolfin``-backed
    ``finmag.util.helpers`` module. [Claude Opus 4.8]
    """
    return re.sub(r"[^a-zA-Z0-9_]", "_", filename)


def save_ndt(sim):
    """Save the average field values (magnetisation etc.) to the .ndt file.

    The filename is derived from the simulation name (as given when the
    simulation was initialised) and has the extension ``.ndt``.
    """
    log.debug("Saving data to ndt file at t={} (sim.name={}).".format(
        sim.t, sim.name))
    sim.tablewriter.save()


def save_m(sim, filename=None, incremental=False, overwrite=False):
    """Convenience wrapper: save the magnetisation to a .npy file."""
    sim.save_field(
        'm', filename=filename, incremental=incremental, overwrite=overwrite)


def create_backup_file_if_file_exists(filename, backupextension='.backup'):
    if os.path.exists(filename):
        backup_file_name = filename + backupextension
        shutil.copy(filename, backup_file_name)
        log.extremedebug("Creating backup %s of %s" %
                         (backup_file_name, filename))


def canonical_restart_filename(sim):
    return sim.sanitized_name + "-restart.npz"


def create_non_existing_parent_directories(filename):
    """Create parent directories of the given file if they do not exist."""
    dirname = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(dirname):
        log.debug(
            "Creating non-existing parent directory: '{}'".format(dirname))
        os.makedirs(dirname)


def mesh_coordinate_hash(coordinates):
    """Return a stable hash of an owned coordinate table.

    The coordinates are sorted and rounded first so the hash is invariant to
    vertex/dof reordering of the same geometric mesh (a serial-only identity
    check; see the serial-only scope of the port).
    """
    rounded = np.round(np.asarray(coordinates, dtype=np.float64), _COORD_DECIMALS)
    order = np.lexsort(rounded.T[::-1])
    return hashlib.sha1(rounded[order].tobytes()).hexdigest()


def save_restart_data(sim, filename=None):
    """Save the current magnetisation, time and metadata to an npz restart file.

    Stored (coordinate-aware format, version 2):

    - ``coordinates`` : owned mesh-vertex coordinates ``(n, gdim)``;
    - ``m``           : coordinate-ordered magnetisation ``(n, 3)`` matching
      ``coordinates`` row-for-row;
    - ``mesh_hash``   : hash of the sorted coordinates (mesh identity check);
    - ``Ms``/``alpha``/``gamma``/``unit_length`` : material parameters;
    - ``interactions`` : sorted list of interaction names present;
    - ``simtime``/``datetime``/``simname``/``driver``/``format_version``.
    """
    datetimetuple = datetime.now()
    drivertype = 'scipy'  # the ported DOLFINx default backend
    simtime = sim.t

    if filename is None:
        filename = canonical_restart_filename(sim)

    create_backup_file_if_file_exists(filename)
    create_non_existing_parent_directories(filename)

    coordinates, values = sim.m_field.coords_and_values()

    np.savez_compressed(
        filename,
        coordinates=np.asarray(coordinates, dtype=np.float64),
        m=np.asarray(values, dtype=np.float64),
        mesh_hash=mesh_coordinate_hash(coordinates),
        # ``Ms``/``alpha`` are lumped to a single representative scalar here
        # (``Ms_av`` is already a volume average; ``sim.alpha`` is a float
        # when uniform, else a per-node array, so ``mean()`` collapses a
        # spatially varying field to one number). Both are informational
        # restart metadata only: ``load_restart_data``/
        # ``apply_restart_magnetisation`` never validate or reapply them
        # against the loaded simulation (only ``m`` is actually restored;
        # see the Task 12 review's accepted behaviour and
        # ``transition-notes.org``'s Task 12 restart section). [Claude
        # Sonnet 5]
        Ms=float(sim.llg.Ms_av),
        alpha=float(np.mean(np.atleast_1d(sim.alpha))),
        gamma=float(sim.gamma),
        unit_length=float(sim.unit_length),
        interactions=np.array(sim.interactions(), dtype=object),
        simtime=simtime,
        datetime=datetimetuple,
        simname=sim.name,
        driver=drivertype,
        format_version=RESTART_FORMAT_VERSION,
    )
    log.debug("Have saved restart data at t=%g to %s (sim.name=%s)" % (
        sim.t, filename, sim.name))


def load_restart_data(filename_or_simulation):
    """Load restart data from an npz file (or a simulation's canonical name).

    Returns a dictionary with the decoded scalar metadata and the coordinate /
    magnetisation arrays.
    """
    if isinstance(filename_or_simulation, finmag.Simulation):
        filename = canonical_restart_filename(filename_or_simulation)
    elif isinstance(filename_or_simulation, str):
        filename = filename_or_simulation
    else:
        raise ValueError(
            "Can only deal with simulations or filenames, but not '%s'" %
            type(filename_or_simulation))

    data = np.load(filename, allow_pickle=True, encoding='bytes')

    def _decode_legacy_bytes(value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, dict):
            return {
                _decode_legacy_bytes(key): _decode_legacy_bytes(val)
                for key, val in value.items()
            }
        if isinstance(value, (list, tuple)):
            return type(value)(_decode_legacy_bytes(item) for item in value)
        return value

    scalar_keys = (
        'simtime', 'simname', 'driver', 'datetime', 'mesh_hash',
        'Ms', 'alpha', 'gamma', 'unit_length', 'format_version',
    )
    result = {}
    for key in data.keys():
        if key in scalar_keys:
            result[key] = _decode_legacy_bytes(data[key].tolist())
        elif key == 'interactions':
            result[key] = [_decode_legacy_bytes(x) for x in data[key].tolist()]
        else:
            result[key] = data[key]

    if 'format_version' not in result or 'coordinates' not in result:
        raise ValueError(
            "'{}' is a legacy v1 raw-dof restart file (no 'format_version' / "
            "'coordinates' keys): this DOLFINx port only supports the "
            "coordinate-aware v2 restart format. A v1 file cannot be safely "
            "remapped by coordinate and is not supported here. Regenerate the "
            "restart file from a fresh run of this (DOLFINx) code, or re-save "
            "the legacy state under the legacy finmag codebase and convert it "
            "before loading.".format(filename))
    if result['format_version'] != RESTART_FORMAT_VERSION:
        raise ValueError(
            "'{}' has restart format_version={!r}, but this DOLFINx port only "
            "supports format_version={} (the coordinate-aware v2 layout). "
            "Regenerate the restart file with the current code.".format(
                filename, result['format_version'], RESTART_FORMAT_VERSION))

    return result


def apply_restart_magnetisation(m_field, data):
    """Set ``m_field`` from restart ``data``, remapping by coordinate.

    Raises ``ValueError`` if the target field's owned coordinates cannot be
    matched one-to-one to the stored coordinates (loud rejection, never a silent
    misassignment).
    """
    stored_coords = np.round(
        np.asarray(data['coordinates'], dtype=np.float64), _COORD_DECIMALS)
    stored_values = np.asarray(data['m'], dtype=np.float64)

    target_coords, _ = m_field.coords_and_values()
    target_coords = np.round(
        np.asarray(target_coords, dtype=np.float64), _COORD_DECIMALS)

    if stored_coords.shape != target_coords.shape:
        raise ValueError(
            "restart mesh mismatch: stored {} coordinates but the target field "
            "has {}".format(stored_coords.shape, target_coords.shape))

    lookup = {tuple(coord): index for index, coord in enumerate(stored_coords)}
    if len(lookup) != stored_coords.shape[0]:
        raise ValueError("restart file has duplicate vertex coordinates")

    remapped = np.empty_like(stored_values)
    for target_index, coord in enumerate(target_coords):
        try:
            source_index = lookup[tuple(coord)]
        except KeyError:
            raise ValueError(
                "restart mesh mismatch: target vertex {} is absent from the "
                "restart file".format(coord.tolist()))
        remapped[target_index] = stored_values[source_index]

    m_field.set_with_ordered_numpy_array_xyz(remapped.reshape(-1))
    return m_field


def eta(sim, when_started):
    """Estimated time of simulation completion (used with run_until)."""
    import time

    elapsed_real_time = time.time() - when_started
    simulation_speed = sim.t / elapsed_real_time
    if simulation_speed > 0 and sim.t < sim.t_max:
        remaining_simulation_time = sim.t_max - sim.t
        remaining_real_time = remaining_simulation_time / simulation_speed
        hours, remainder = divmod(remaining_real_time, 60)
        minutes, seconds = divmod(remainder, 60)
        log.info("Integrated up to t = {:.4} ns. "
                 "Predicted end in {:0>2}:{:0>2}:{:0>2}.".format(
                     sim.t * 1e9, int(hours), int(minutes), int(seconds)))


# -- Task 15: relax()/hysteresis() pure-Python helpers -----------------------
#
# Reimplemented locally (identical to their ``finmag.util.helpers``
# counterparts) so ``sim_relax``/``hysteresis`` do not import the legacy
# ``dolfin``-backed ``finmag.util.helpers`` module, matching the
# ``clean_filename`` precedent above. [Claude Sonnet 5]

def norm(vs):
    """Euclidian norm of one or several 3-vectors.

    When passing an array of vectors, the shape is expected to be
    ``[[x0, y0, z0], ..., [xn, yn, zn]]``.
    """
    if not type(vs) == np.ndarray:
        vs = np.array(vs)
    if vs.shape == (3,):
        return np.linalg.norm(vs)
    return np.sqrt(np.add.reduce(vs * vs, axis=1))


def compute_dmdt(t0, m0, t1, m1):
    """Maximum of the L2 norm of dm/dt between two magnetisation snapshots.

    Arguments:
        t0, t1: two points in time (floats)
        m0, m1: the magnetisation at t0, resp. t1 (np.arrays of shape 3*n)
    """
    dm = (m1 - m0).reshape((3, -1))
    max_dm = np.max(np.sqrt(np.sum(dm ** 2, axis=0)))  # max of L2-norm
    dt = abs(t1 - t0)
    return max_dm / dt
