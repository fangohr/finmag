"""DOLFINx direct port of the analytic thin-film demagnetising approximation.

Task 19 ports the last legacy-tested optional energy still importing
``dolfin`` at module scope: ``ThinFilmDemag`` is a closed-form approximation
for thin films magnetised close to a single axis (``direction``, default
``"z"``) -- ``Hj = Hk = 0`` and ``Hi = -strength_i * m_i`` for the chosen
axis ``i``, with no genuine BEM/Poisson solve involved. It is not an
``EnergyBase`` subclass in legacy either (it hard-codes ``compute_energy() ==
0`` -- no energy functional is associated with this field-only
approximation), so this port mirrors legacy's own bespoke, non-``EnergyBase``
structure exactly, the same way :mod:`finmag.energies.zeeman` does for its
non-``EnergyBase`` interactions.

Ported convention (documented deviation, shared with every other ported
interaction): :meth:`compute_field` returns the flat owned array in DOLFINx's
own interleaved-per-node dof order (matching
:meth:`finmag.field.Field.as_array`), not legacy's component-major
``(3, n).ravel()`` layout. The physical per-(node, component) values are
identical; only the serialisation order changes, exactly like every other
ported energy's ``compute_field()``. [Claude Sonnet 5]
"""

import logging

import numpy as np
from ufl import TestFunction, dx

from finmag.field import Field, associated_scalar_space, owned_raw_to_blocked

from .energy_base import _assemble_vector_owned, _nodal_volume_owned, _require_cg1_magnetisation

log = logging.getLogger(name="finmag")

_DIRECTION_INDEX = {"x": 0, "y": 1, "z": 2}


class ThinFilmDemag:
    """Demagnetising field for thin films in the ``direction`` axis.

    ``Hj = Hk = 0`` and ``Hi = -strength_i * m_i`` for the chosen axis ``i``
    (default ``"z"``): valid only for films thin and magnetised close to that
    axis's plane normal. ``field_strength`` is ``Ms`` by default (per-node,
    box-averaged exactly like :class:`finmag.energies.energy_base.EnergyBase`
    -- see :meth:`setup`); passing an explicit ``field_strength`` bypasses the
    ``Ms``-average branch entirely and is used verbatim (legacy applies no
    validation or coercion to a user-supplied ``field_strength`` either;
    transcribed as-is).
    """

    def __init__(self, direction="z", field_strength=None, in_jacobian=False,
                 name="ThinFilmDemag"):
        """``field_strength`` is ``Ms`` by default."""
        if direction not in ("x", "y", "z"):
            raise ValueError("direction must be one of 'x', 'y', 'z'")
        self.direction_label = direction
        self.direction = _DIRECTION_INDEX[direction]
        self.strength = field_strength
        self.in_jacobian = in_jacobian
        self.name = name
        in_jacobian_msg = "in Jacobian" if in_jacobian else "not in Jacobian"
        log.debug("Creating {} object, {}.".format(
            self.__class__.__name__, in_jacobian_msg))

    def setup(self, m, Ms, unit_length):
        if not isinstance(m, Field):
            raise TypeError("m must be a finmag.Field")
        if not isinstance(Ms, Field):
            raise TypeError("Ms must be a finmag.Field")
        if m.is_scalar_field() or m.value_dim() != 3:
            raise ValueError("ThinFilmDemag requires a three-component m Field")
        _require_cg1_magnetisation(m)
        if not Ms.is_scalar_field():
            raise ValueError("Ms must be a scalar Field")
        if m.mesh() is not Ms.mesh():
            raise ValueError("m and Ms must use the same mesh")

        self.m = m
        self.Ms = Ms
        self.unit_length = float(unit_length)
        n_owned_nodes = m.functionspace.dofmap.index_map.size_local
        self.H = np.zeros((n_owned_nodes, 3))

        if self.strength is None:
            # Box-assemble the (possibly non-uniform) Ms Field onto m's
            # associated CG1 scalar space, exactly like legacy's own
            # ``df.assemble(Ms.f * TestFunction(S1) * dx).get_local() /
            # volumes.get_local()`` -- the same lumped nodal-average math
            # EnergyBase's box-assemble path uses elsewhere in this port.
            self.S1 = associated_scalar_space(m.functionspace)
            scalar_test = TestFunction(self.S1)
            nodal_ms = _assemble_vector_owned(Ms.f * scalar_test * dx, self.S1)
            nodal_volume = _nodal_volume_owned(self.S1)
            self.strength = nodal_ms / nodal_volume
        return self

    def compute_field(self):
        """Collectively return the field in legacy component-blocked order.

        Recomputed fresh from the currently bound ``m`` on every call (like
        legacy), so it reflects the live magnetisation state, not a value
        cached at :meth:`setup` time. Public field-array surface (Task 31):
        the raw node-interleaved owned array is converted once to the legacy
        component-blocked, owned-vertex ``xxx`` view via the shared helper.
        """
        m_values = self.m.as_array().reshape((-1, 3))
        self.H[:, self.direction] = -self.strength * m_values[:, self.direction]
        return owned_raw_to_blocked(self.m.functionspace, self.H.reshape(-1))

    def compute_energy(self):
        """Legacy hard-codes a literal ``0`` here: this field-only
        approximation has no associated energy functional. Transcribed
        verbatim -- not invented -- rather than assembling a form that
        legacy itself never defined."""
        return 0
