"""DOLFINx static and time-dependent Zeeman interactions.

Task 15 ports the whole legacy Zeeman family (``TimeZeeman``,
``DiscreteTimeZeeman``, ``TimeZeemanPython``, ``OscillatingZeeman`` and
``DipolarField``) directly on top of the static :class:`Zeeman` port. None of
these use a dolfin ``Expression``-with-mutable-``.t``-attribute -- DOLFINx has
no such object -- so every one of them is transcribed onto the closest
DOLFINx-native contract instead; see each class's docstring and
``transition-notes.org`` (Task 15) for the exact input-type deviation. Legacy
behavior (including two documented legacy quirks -- ``t_off=0.0`` being
falsy-disabled, and ``DiscreteTimeZeeman`` never advancing its update clock)
is preserved bit-for-bit rather than "fixed". [Claude Sonnet 5]
"""

import logging
from math import cos, pi

import numpy as np
from dolfinx import fem
from mpi4py import MPI
from ufl import dx, inner

from finmag.field import Field, associated_scalar_space

from .energy_base import (
    _assemble_scalar,
    _require_cg1_magnetisation,
    mu0,
)

log = logging.getLogger(name="finmag")


class Zeeman:
    """Static external field in A/m.

    Constants and Python callables use the same assignment contract as
    :class:`finmag.field.Field`. Energy assembly and averages are collective;
    ``compute_field`` returns flat rank-local owned backend-order values.
    """

    def __init__(self, H, name="Zeeman", **kwargs):
        self.H_value = H
        self.name = name
        self.kwargs = kwargs
        self.in_jacobian = False

    def setup(self, m, Ms, unit_length=1.0):
        if not isinstance(m, Field):
            raise TypeError("m must be a finmag.Field")
        if not isinstance(Ms, Field):
            raise TypeError("Ms must be a finmag.Field")
        if m.is_scalar_field() or m.value_dim() != 3:
            raise ValueError("Zeeman requires a three-component m Field")
        _require_cg1_magnetisation(m)
        if not Ms.is_scalar_field():
            raise ValueError("Ms must be a scalar Field")
        if m.mesh() is not Ms.mesh():
            raise ValueError("m and Ms must use the same mesh")

        unit_length = float(unit_length)
        if not np.isfinite(unit_length) or unit_length <= 0.0:
            raise ValueError("unit_length must be a positive finite number")
        ms_values = Ms.as_array()
        local_invalid_ms = bool(
            np.any(~np.isfinite(ms_values)) or np.any(ms_values <= 0.0)
        )
        if m.mesh().comm.allreduce(local_invalid_ms, op=MPI.LOR):
            raise ValueError("Ms must be positive")
        # Spatially varying Ms supported (Task 16): Ms.f enters the energy
        # density verbatim, exactly as legacy used it.

        self.m = m
        self.Ms = Ms
        self.unit_length = unit_length
        self.S1 = associated_scalar_space(m.functionspace)
        if hasattr(self, "H"):
            del self.H
        if hasattr(self, "_energy_density_field"):
            del self._energy_density_field
        self.set_value(self.H_value, **self.kwargs)
        return self

    def set_value(self, value, **kwargs):
        """Set a constant or callable external field after ``setup``."""
        if not hasattr(self, "m"):
            raise RuntimeError("Zeeman.setup must be called before set_value")
        if kwargs:
            raise NotImplementedError(
                "legacy Expression parameters are unavailable; pass a callable"
            )
        if hasattr(self, "H"):
            self.H.set(value)
        else:
            self.H = Field(self.m.functionspace, value, name="H_ext")
        self.H_value = value
        self.value = value
        self.E = -mu0 * self.Ms.f * inner(self.m.f, self.H.f)
        return self

    def compute_field(self):
        """Return the applied field in legacy component-blocked (``xxx``) order.

        Public field-array surface (Task 31): the external field stored in
        ``self.H`` is returned in the legacy component-blocked, owned-vertex
        ordering (identical to ``self.H.get_ordered_numpy_array_xxx()``); the
        raw backend-order dofs remain available via ``self.H.as_array()``.
        The whole time/dipolar Zeeman family inherits this method.
        """
        return self.H.get_ordered_numpy_array_xxx()

    def average_field(self):
        """Collectively return the legacy arithmetic nodal field average."""
        values = self.H.as_array().reshape((-1, self.m.value_dim()))
        local_sum = np.sum(values, axis=0)
        global_sum = np.zeros_like(local_sum)
        self.m.mesh().comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
        global_count = self.m.mesh().comm.allreduce(values.shape[0], op=MPI.SUM)
        return global_sum / global_count

    def compute_energy(self, dx=dx):
        """Collectively integrate Zeeman energy over the supplied measure."""
        mesh_energy = _assemble_scalar(self.m.mesh(), self.E * dx)
        return mesh_energy * self.unit_length ** self.m.mesh_dim()

    def energy_density(self):
        """Collectively return legacy pointwise nodal density as a Field."""
        if self.Ms.is_constant():
            ms_nodal = self.Ms.as_constant()
        else:
            # Interpolate a (possibly DG0) Ms onto the CG1 nodal S1 space so
            # the pointwise density has one Ms per magnetisation node.
            ms_field = Field(self.S1)
            ms_field.from_field(self.Ms)
            ms_nodal = ms_field.as_array()
        values = -mu0 * ms_nodal * np.sum(
            self.m.as_array().reshape((-1, 3))
            * self.H.as_array().reshape((-1, 3)),
            axis=1,
        )
        expected_size = self.S1.dofmap.index_map.size_local
        if values.shape != (expected_size,):
            raise ValueError(
                "Zeeman pointwise density requires matching scalar/vector "
                "CG1 ownership"
            )
        if not hasattr(self, "_energy_density_field"):
            self._energy_density_field = Field(
                self.S1, name="{}_energy_density".format(self.name)
            )
        self._energy_density_field.from_array(values)
        return self._energy_density_field

    def energy_density_function(self):
        return self.energy_density().f


def _bind_zeeman_fields(interaction, m, Ms, unit_length):
    """Validate and bind ``m``/``Ms``/``unit_length``/``S1`` onto ``interaction``.

    Transcribed from :meth:`Zeeman.setup`'s validation block (kept private and
    duplicated rather than factored into ``Zeeman`` itself, so the ported
    static ``Zeeman`` class stays byte-for-byte untouched by this slice).
    Shared by :class:`TimeZeemanPython`, whose ``setup`` differs from
    ``Zeeman.setup`` only in how the ``H`` field is first populated.
    """
    if not isinstance(m, Field):
        raise TypeError("m must be a finmag.Field")
    if not isinstance(Ms, Field):
        raise TypeError("Ms must be a finmag.Field")
    if m.is_scalar_field() or m.value_dim() != 3:
        raise ValueError(
            "{} requires a three-component m Field".format(
                type(interaction).__name__
            )
        )
    _require_cg1_magnetisation(m)
    if not Ms.is_scalar_field():
        raise ValueError("Ms must be a scalar Field")
    if m.mesh() is not Ms.mesh():
        raise ValueError("m and Ms must use the same mesh")

    unit_length = float(unit_length)
    if not np.isfinite(unit_length) or unit_length <= 0.0:
        raise ValueError("unit_length must be a positive finite number")
    ms_values = Ms.as_array()
    local_invalid_ms = bool(
        np.any(~np.isfinite(ms_values)) or np.any(ms_values <= 0.0)
    )
    if m.mesh().comm.allreduce(local_invalid_ms, op=MPI.LOR):
        raise ValueError("Ms must be positive")
    # Spatially varying Ms supported (Task 16); Ms.f used verbatim.

    interaction.m = m
    interaction.Ms = Ms
    interaction.unit_length = unit_length
    interaction.S1 = associated_scalar_space(m.functionspace)


def _as_time_field_function(field_expression, t_off, cls_name):
    """Normalise the legacy ``field_expression`` argument into ``f(t) -> value``.

    Ported input contract (deliberate deviation, documented in
    ``transition-notes.org``): the legacy class accepted a ``dolfin``
    ``Expression`` with a mutable ``.t`` attribute mutated by ``update(t)``.
    DOLFINx has no such object, so the ported contract is a plain Python
    callable ``field_function(t)`` returning any value
    :meth:`finmag.field.Field.set` accepts -- a constant 3-vector/array, a
    per-point or vectorized callable ``g(x)``, a ``dolfinx.fem.Function``, or
    another :class:`~finmag.field.Field` -- describing the field at time
    ``t``. This preserves the full legacy capability (spatially *and*
    temporally varying fields) without the Expression string mechanism.

    A constant 3-array/tuple/list is still accepted directly, matching
    legacy's safety check: it is only meaningful together with ``t_off``
    (otherwise there would be no time update at all), and is wrapped into a
    callable that always returns that same constant.
    """
    if callable(field_expression):
        return field_expression
    value = np.asarray(field_expression, dtype=np.float64)
    if value.shape != (3,):
        raise ValueError(
            "If field_expression is not callable, it must be a 3-array "
            "(representing a constant external field)"
        )
    if t_off is None:
        raise ValueError(
            "The argument 'field_expression' is a constant array, but "
            "t_off was not specified so there will be no time update "
            "at all. Use the Zeeman class instead of {} if this is what "
            "you really want.".format(cls_name)
        )
    constant = tuple(float(v) for v in value)
    return lambda t: constant


class TimeZeeman(Zeeman):
    """Time-dependent external field (in A/m), updated as continuously as
    possible.

    See :func:`_as_time_field_function` for the ported ``field_expression``
    input contract. ``t_off`` switches the field off (zeroed, interaction
    kept -- matching the ``sim.switch_off_H_ext`` zero-but-keep precedent from
    Task 9): once ``self.switched_off`` becomes ``True`` it stays ``True``.

    Preserved legacy quirk: the switch-off test is ``if self.t_off and t >=
    self.t_off`` (truthiness, not ``is not None``), so ``t_off=0.0`` is
    silently *never* triggered (0.0 is falsy). This is transcribed verbatim
    rather than "fixed"; pass a small positive ``t_off`` to switch off at
    (approximately) t=0 if that is the intent.
    """

    def __init__(self, field_expression, t_off=None, name="TimeZeeman"):
        self.field_function = _as_time_field_function(
            field_expression, t_off, type(self).__name__
        )
        self.t_off = t_off
        self.switched_off = False
        super().__init__(self.field_function(0.0), name=name)

    def update(self, t):
        if self.switched_off:
            return
        if self.t_off and t >= self.t_off:
            self.switch_off()
            return
        self.set_value(self.field_function(t))

    def switch_off(self):
        log.debug("Switching external field off.")
        self.set_value((0.0, 0.0, 0.0))
        self.switched_off = True


class DiscreteTimeZeeman(TimeZeeman):
    """Time-dependent external field, updated at discrete time intervals.

    ``update(t)`` refreshes the field only once at least ``dt_update`` time
    has passed since the last refresh (``dt_update=None`` together with
    ``t_off`` gives a field that stays constant until it is switched off).

    Preserved legacy quirks (transcribed verbatim, not fixed --
    see ``transition-notes.org``):

    1. legacy's ``update`` never advances ``self.t_last_update`` after the
       first refresh, so the interval check ``t - self.t_last_update >=
       self.dt_update`` keeps comparing against the ``__init__``-time value
       of ``0.0`` forever. In practice this means the field is held at its
       initial value until ``t`` first reaches ``dt_update``, and from then
       on every subsequent ``update(t)`` call refreshes the field again
       (rather than only every ``dt_update`` thereafter).
    2. legacy's ``update`` refreshes ``self.H`` by assigning a *brand-new*
       ``Field`` object directly (bypassing ``set_value()``, which the base
       ``TimeZeeman.update`` uses and which also rebuilds the cached energy
       form ``self.E``). Because ``self.E`` was only ever built once, in
       ``setup()``, against the *original* ``H`` ``Function`` object,
       ``compute_energy()`` silently keeps returning the energy of the very
       first (setup-time) field value forever after the first interval
       update, even though ``compute_field()``/``energy_density()`` (which
       read ``self.H``'s current array directly) correctly reflect the
       updated field. This field/energy inconsistency was discovered while
       building the Task 15 oracle fixture (see
       ``fixtures/timezeeman_oracle.json``'s ``discrete_time_zeeman`` case,
       where ``energy`` stays exactly the setup-time value across every
       snapshot even as ``H_vertex`` visibly changes) and is a genuine,
       previously-undocumented legacy defect. It is transcribed verbatim
       here (DELIBERATE PRESERVATION, USER ACCEPTANCE PENDING) rather than
       silently fixed.
    """

    def __init__(self, field_expression, dt_update=None, t_off=None,
                 name="DiscreteTimeZeeman"):
        if dt_update is None and t_off is None:
            raise ValueError(
                "At least one of the arguments 'dt_update' and 't_off' "
                "must be given."
            )
        super().__init__(field_expression, t_off, name=name)
        self.dt_update = dt_update
        self.t_last_update = 0.0

    def update(self, t):
        if self.switched_off:
            return
        if self.t_off and t >= self.t_off:
            self.switch_off()
            return
        if self.dt_update is not None:
            dt_since_last_update = t - self.t_last_update
            if dt_since_last_update >= self.dt_update:
                # Preserved legacy quirk 2 (see class docstring): assigns a
                # brand-new Field to self.H directly, like legacy's own
                # ``self.H = Field(..., self.value, name='H_ext')`` --
                # *not* ``self.set_value(...)`` -- so the cached ``self.E``
                # energy form (built once in ``setup()``) is never rebuilt.
                self.H = Field(
                    self.m.functionspace, self.field_function(t),
                    name="H_ext")
                log.debug(
                    "At t={}, after dt={}, update external field "
                    "again.".format(t, dt_since_last_update)
                )
                # NOTE: legacy does *not* set self.t_last_update = t here;
                # preserved verbatim, see the class docstring (quirk 1).


class TimeZeemanPython(TimeZeeman):
    """Faster time-dependent field for the case that only a scalar amplitude
    varies over time:

        H(t, x) = H0(x) * time_fun(t)

    Ported input contract: ``H0_value`` is any value accepted by
    :meth:`finmag.field.Field.set` (constant vector, per-point/vectorized
    callable, ``dolfinx.fem.Function``, or ``Field``) describing the
    time-*independent* spatial field; ``time_fun(t)`` returns a scalar
    amplitude. This mirrors legacy's documented common case (a vector-valued
    spatial expression combined with a scalar ``time_fun``) and is the exact
    case :class:`OscillatingZeeman` uses. The legacy scalar-spatial-envelope-
    with-vector-``time_fun`` branch (one shared scalar envelope multiplied
    independently per vector component, e.g. for a rotating field) is not
    exercised by any legacy test and is not ported in this slice -- see
    ``transition-notes.org``.

    ``H0`` is interpolated once in :meth:`setup`; every subsequent
    :meth:`update` only rescales the cached array, matching the legacy
    performance rationale exactly.

    By-name gate (Task 15 deviation, not a generic crash): :meth:`setup`
    probes ``time_fun(0.0)`` and raises :class:`NotImplementedError` naming
    the unported "TimeZeemanPython vector-valued time_fun" branch if it is
    not scalar-like. Without this gate, the vector-``time_fun`` branch would
    instead fail with a generic ``ValueError`` from :meth:`Field.set`
    rejecting an incompatible pointwise shape, or a raw ``TypeError`` from
    ``float(self.time_fun(t))`` first hit mid-integration in :meth:`_apply`.
    """

    def __init__(self, H0_value, time_fun, t_off=None, name="TimeZeemanPython"):
        self.H0_value = H0_value
        self.time_fun = time_fun
        self.t_off = t_off
        self.switched_off = False
        self.name = name
        self.in_jacobian = False

    def setup(self, m, Ms, unit_length=1.0):
        _bind_zeeman_fields(self, m, Ms, unit_length)
        self._check_time_fun_is_scalar_valued()

        H0_field = Field(m.functionspace, name="H0")
        H0_field.set(self.H0_value)
        self._H_init = H0_field.as_array().copy()

        if hasattr(self, "H"):
            del self.H
        if hasattr(self, "_energy_density_field"):
            del self._energy_density_field
        self.H = Field(m.functionspace, name="H_ext")
        self.switched_off = False
        self._apply(0.0)
        self.E = -mu0 * self.Ms.f * inner(self.m.f, self.H.f)
        return self

    def _check_time_fun_is_scalar_valued(self):
        """By-name gate for the unported vector-valued ``time_fun`` branch.

        Probes ``time_fun(0.0)``: the legacy scalar-spatial-envelope-with-
        vector-``time_fun`` branch (one shared scalar envelope multiplied
        independently per vector component, e.g. for a rotating field) is
        not exercised by any legacy test and is not ported in this slice
        (see the class docstring and ``transition-notes.org``/
        ``dev/dolfinx/porting_map.md`` Task 15). Without this upfront,
        by-name check, that branch would instead fail late with a generic
        ``ValueError`` from ``Field.set`` (an incompatible pointwise value
        shape) at ``setup``, or a raw ``TypeError`` from
        ``float(self.time_fun(t))`` the first time :meth:`_apply` runs
        mid-integration.
        """
        probe = np.asarray(self.time_fun(0.0))
        if probe.shape not in ((), (1,)):
            raise NotImplementedError(
                "TimeZeemanPython vector-valued time_fun is not ported "
                "(Task 15 deviation): time_fun(t) must return a scalar "
                "amplitude, not a {}-shaped value. The legacy scalar-"
                "spatial-envelope-with-vector-time_fun branch "
                "(independently scaling each component of a shared scalar "
                "spatial envelope, e.g. for a rotating field) is not "
                "exercised by any legacy test and is not ported in this "
                "slice.".format(probe.shape)
            )

    def _apply(self, t):
        scale = float(self.time_fun(t))
        self.H.from_array(self._H_init * scale)

    def update(self, t):
        if self.switched_off:
            return
        if self.t_off and t >= self.t_off:
            self.switch_off()
            return
        self._apply(t)

    def switch_off(self):
        log.debug("Switching external field off.")
        self.H.from_array(np.zeros_like(self._H_init))
        self.switched_off = True


class OscillatingZeeman(TimeZeemanPython):
    """Field constant in space, oscillating sinusoidally in time:

        H(t) = H0 * cos(2*pi*freq*t + phase)

    Transcribed verbatim from the legacy formula: ordinary (not angular)
    frequency, cosine (not sine), phase added inside the argument (so the
    value at t=0 is ``H0 * cos(phase)``).
    """

    def __init__(self, H0, freq, phase=0, t_off=None, name="OscillatingZeeman"):
        H0_value = tuple(float(v) for v in H0)

        def amplitude(t):
            return cos(2.0 * pi * freq * t + phase)

        super().__init__(H0_value, amplitude, t_off=t_off, name=name)


class DipolarField(Zeeman):
    """Magnetostatic field of a point dipole at position ``pos`` with a fixed
    magnetic moment.

    If ``magnitude`` is ``None``, the magnetic moment is simply given by
    ``m``. Otherwise ``m`` is interpreted only as the *direction* of the
    magnetic moment and ``magnitude`` as its magnitude, i.e. the magnetic
    moment is ``magnitude * (m / |m|)``.

    Ported input contract (deliberate deviation): legacy built a ``dolfin``
    ``Expression`` string evaluating this same closed-form point-dipole field,
    although the class also defined (but never used -- it is commented out
    in the legacy source) a plain per-point Python closure, ``H_fun``,
    computing the identical formula. DOLFINx has no Expression equivalent, so
    this port uses exactly that closed-form callable directly, vectorised
    over :class:`~finmag.field.Field`'s callable-interpolation contract.
    [Claude Sonnet 5]
    """

    def __init__(self, pos, m, magnitude=None, name="DipolarField"):
        self.pos = np.asarray(pos, dtype=np.float64)
        if magnitude is None:
            self.m = np.asarray(m, dtype=np.float64)
        else:
            self.m = magnitude * np.asarray(m, dtype=np.float64) / np.linalg.norm(m)

        pos = self.pos
        moment = self.m

        def H_fun(pt):
            pt = np.asarray(pt, dtype=np.float64)
            if pt.ndim == 1:
                v = pos - pt
                r = np.linalg.norm(v)
                return 1.0 / (4 * pi) * (
                    3 * v * np.dot(moment, v) / r ** 5 - moment / r ** 3
                )
            v = pos[:, None] - pt
            r = np.linalg.norm(v, axis=0)
            dotted = np.einsum("i,ij->j", moment, v)
            return 1.0 / (4 * pi) * (
                3 * v * dotted / r ** 5 - moment[:, None] / r ** 3
            )

        super().__init__(H_fun, name=name)
