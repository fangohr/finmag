"""Dolfin-free readers for checked-in Magpar reference data.

The legacy :mod:`finmag.util.magpar` module imports ``dolfin`` at module scope
(and its writer/live-run helpers genuinely need it), so it cannot be imported in
the DOLFINx environment. The *readers*, however -- ``read_femsh``, ``read_inp``,
``get_field`` and ``get_m0`` -- are pure NumPy and only parse the checked-in
Magpar output files (the ``.femsh`` node table and the gzip-ed ``.0001`` AVS
field table). They are extracted here verbatim so DOLFINx comparison tests can
read Magpar reference data (fields sampled at *saved coordinates*) without any
legacy dependency.

This module contains no ``dolfin`` import and no live-Magpar execution: running
Magpar itself (``magpar.exe``) remains deferred (acceptance register M15). The
functions below are byte-for-byte faithful to
``git show b5015c5a:src/finmag/util/magpar.py`` so that any reference read is
identical to the legacy read. [Claude Opus 4.8]
"""

import gzip
import os

import numpy as np


def read_femsh(file_name):
    """Read a Magpar ``.femsh`` mesh file.

    Returns ``(node_coord, connectivity)`` where ``node_coord`` has shape
    ``(n_node, 3)`` and ``connectivity`` has shape ``(n_cell, 4)``.
    """
    f = open(file_name, 'r')
    a = f.readline()
    n_node = int(a.split()[0])
    n_cell = int(a.split()[1])

    node_coord = []
    for i in range(n_node):
        line = f.readline()
        tmp = line.split()
        t2 = [float(tmp[1]), float(tmp[2]), float(tmp[3])]
        node_coord.append(t2)

    connectivity = []
    for i in range(n_cell):
        t = f.readline().split()
        t2 = [int(t[3]), int(t[4]), int(t[5]), int(t[6])]
        connectivity.append(t2)

    return np.array(node_coord), np.array(connectivity)


def read_inp(file_name):
    """Read a Magpar AVS ``.inp`` field table (plain or gzip-ed ``.gz``).

    Returns a dict mapping each column name (e.g. ``"Hexch_x"``) to a flat
    NumPy array of node values.
    """
    # magpar produces gzip-ed files, so look for file_name.gz as well as file_name
    if os.path.isfile(file_name):
        f = open(file_name, 'r')
    elif os.path.isfile(file_name + '.gz'):
        f = gzip.open(file_name + '.gz', 'rt')
    else:
        raise OSError("No such file: %s" % file_name)

    with f:
        a = f.readline()
        num = int(a.split()[0])
        names = []
        for i in range(num):
            names.append(f.readline().split(',')[0])

        lines = f.readlines()

    data = []
    for line in lines:
        data.append([float(t) for t in line.split()])

    data = np.array(data)

    fields = {}
    for i in range(num):
        fields[names[i]] = data[:, i + 1]

    return fields


def get_field(base_name, field="anis"):
    """
    Read values for the given field from file with the given
    base name.  Returns a pair `(nodes, field_vals)` where
    `nodes` is an array of shape Nx3 containing the coordinates
    of the mesh nodes and `field_vals` is a flat array in
    component-blocked order ``[fx0..fxn, fy0..fyn, fz0..fzn]``.

    The field is converted from Magpar's Tesla convention to A/m by
    dividing by ``mu0 = 4*pi*1e-7``.
    """
    file_name = base_name + ".0001"
    fields = read_inp(file_name)

    if field == "anis":
        fx = fields["Hani_x"]
        fy = fields["Hani_y"]
        fz = fields["Hani_z"]
    elif field == "exch":
        fx = fields["Hexch_x"]
        fy = fields["Hexch_y"]
        fz = fields["Hexch_z"]
    elif field == "demag":
        fx = fields["Hdemag_x"]
        fy = fields["Hdemag_y"]
        fz = fields["Hdemag_z"]
    else:
        raise NotImplementedError(
            "only exch, anis or demag field can be extracted now")

    field = np.array([fx, fy, fz]).reshape(1, -1, order='C')[0]
    field = field / (np.pi * 4e-7)

    file_name = base_name + ".0001.femsh"
    nodes, connectivity = read_femsh(file_name)

    return nodes, field


def get_m0(file_name):
    """Read the initial magnetisation ``M_{x,y,z}`` from a Magpar ``.inp`` file.

    Returns a flat array in component-blocked order ``[Mx.., My.., Mz..]``.
    """
    fields = read_inp(file_name)
    fx = fields["M_x"]
    fy = fields["M_y"]
    fz = fields["M_z"]

    field = np.array([fx, fy, fz]).reshape(1, -1)[0]
    return field
