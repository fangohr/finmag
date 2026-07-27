# FinMag
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk

"""Stdlib+numpy-only coordinate-conversion helpers for the public Finmag
boundary.

``spherical_to_cartesian`` (and its inverse ``cartesian_to_spherical``) are
pure-numpy vector-maths helpers historically hosted in
:mod:`finmag.util.helpers`, which imports legacy ``dolfin`` at module scope.
SR1 S1b lifts them into a module with no FEM dependency at all, following the
``finmag.util.logging_helpers`` precedent (SR1 P2.1/P4-helpers) and the
``finmag.util.plot_helpers`` narrow-sourcing precedent (SR1 P4-viz), so
callers that only need the coordinate conversion (e.g.
``examples/magnetic_grain/suess_2001.py``) can import it without pulling in
legacy ``dolfin``. Copied verbatim from ``helpers.py`` -- neither function
ever touched ``dolfin`` -- and re-exported from ``helpers.py`` so the legacy
spelling ``from finmag.util.helpers import spherical_to_cartesian`` keeps
working. [Claude Sonnet 5]
"""

import numpy as np


def cartesian_to_spherical(vector):
    """
    Converts cartesian coordinates to spherical coordinates.

    Returns a tuple (r, theta, phi) where r is the radial distance, theta
    is the inclination (or elevation) and phi is the azimuth (ISO standard 31-11).

    """
    r = np.linalg.norm(vector)
    unit_vector = np.array(vector) / r
    theta = np.arccos(unit_vector[2])
    phi = np.arctan2(unit_vector[1], unit_vector[0])
    return np.array((r, theta, phi))


def spherical_to_cartesian(v):
    """
    Converts spherical coordinates to cartesian.

    Expects the arguments r for radial distance, inclination theta
    and azimuth phi.

    """
    r, theta, phi = v
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array((x, y, z))
