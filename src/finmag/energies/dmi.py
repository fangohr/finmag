"""DOLFINx Dzyaloshinskii-Moriya interaction (DMI)."""

import numbers

import numpy as np
from aeon import timer
from ufl import curl, grad, inner

from finmag.field import Field

from .energy_base import EnergyBase, _require_cg1_magnetisation
from dolfinx import fem


class DMI(EnergyBase):
    """Compute the Dzyaloshinskii-Moriya interaction (DMI) energy and field.

    .. math::
        E_{\\text{DMI}} = \\int_\\Omega D \\vec{m} \\cdot
                          (\\nabla \\times \\vec{m}) dx

    This first DOLFINx slice supports a spatially constant scalar ``D`` and
    the ``box-assemble`` method (matching the Task 5 energy foundation).

    *Arguments*
        D
            the (constant, scalar) DMI constant.
        method
            only ``'box-assemble'`` is supported; every other legacy method
            keeps raising ``NotImplementedError`` precisely, exactly as for
            :class:`~finmag.energies.exchange.Exchange`.
        dmi_type
            ``'auto'`` (default), ``'1d'``, ``'2d'``, ``'3d'`` or
            ``'interfacial'``. ``'auto'`` dispatches on the mesh dimension,
            matching the legacy ``DMI`` class exactly. The legacy ``'D2D'``
            variant is not ported in this slice and raises
            ``NotImplementedError`` by name.
    """

    _bulk_dim_overrides = {"1d": 1, "2d": 2, "3d": 3}

    def __init__(self, D, method="box-assemble", name="DMI", dmi_type="auto"):
        self.D_value = _constant_scalar_value(D, "D")
        self.name = name
        self.dmi_type = _validate_dmi_type(dmi_type)

        super(DMI, self).__init__(method=method, in_jacobian=True)

    @timer.method
    def setup(self, m, Ms, unit_length=1.0):
        if not isinstance(m, Field):
            raise TypeError("m must be a finmag.Field")
        if m.value_dim() != 3 or m.is_scalar_field():
            raise ValueError("DMI requires a three-component m Field")
        _require_cg1_magnetisation(m)

        unit_length = float(unit_length)
        if not np.isfinite(unit_length) or unit_length <= 0.0:
            raise ValueError("unit_length must be a positive finite number")

        coefficient_space = fem.functionspace(m.mesh(), ("DG", 0))
        self.D = Field(coefficient_space, self.D_value, name="D")

        # Multiplication factor used for the DMI energy computation, exactly
        # matching the legacy ``dmi_factor = df.Constant(1.0 / unit_length)``.
        # Combined with ``EnergyBase``'s ``unit_length**self.dim`` total-energy
        # scaling (``self.dim`` is the actual mesh dimension), this reproduces
        # the legacy ``unit_length ** (dim - 1)`` DMI scaling convention.
        self.dmi_factor = fem.Constant(m.mesh(), 1.0 / unit_length)

        dmi_dim = self._bulk_dim_overrides.get(self.dmi_type, m.mesh_dim())

        if self.dmi_type == "interfacial":
            integrand = _dmi_interfacial(m.f, dmi_dim)
        else:
            integrand = _times_curl(m.f, dmi_dim)

        E_integrand = self.dmi_factor * self.D.f * integrand

        super(DMI, self).setup(E_integrand, m, Ms, unit_length)
        return self


def _constant_scalar_value(value, name):
    if isinstance(value, (Field, fem.Function, str)) or callable(value):
        raise NotImplementedError(
            "spatially varying {} is deferred from the first DOLFINx "
            "DMI slice".format(name)
        )
    if isinstance(value, fem.Constant):
        value = value.value
    if isinstance(value, numbers.Real):
        result = float(value)
    else:
        array = np.asarray(value)
        if array.size != 1:
            raise NotImplementedError(
                "spatially varying {} is deferred from the first DOLFINx "
                "DMI slice".format(name)
            )
        result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _validate_dmi_type(dmi_type):
    known = ("auto", "1d", "2d", "3d", "interfacial")
    if dmi_type == "D2D":
        raise NotImplementedError(
            "dmi_type='D2D' is not yet ported to DOLFINx; supported values "
            "are {}".format(known)
        )
    if dmi_type not in known:
        raise ValueError(
            "unsupported dmi_type {!r}; supported values are {}".format(
                dmi_type, known
            )
        )
    return dmi_type


def _times_curl(m, dim):
    """Return ``m . curl(m)`` (bulk DMI), transcribed from the legacy
    ``finmag.util.helpers.times_curl``.

    On a three-dimensional mesh this is UFL's native ``curl``. On one- and
    two-dimensional meshes ``curl`` is not defined, so the Cartesian
    expansion is used instead, with derivatives that do not exist on the
    mesh (``z`` always; ``y`` when ``dim == 1``) set to zero exactly as in
    the legacy transcription.
    """
    if dim == 3:
        return inner(m, curl(m))

    gradm = grad(m)

    # Derivatives along x exist in both the 1d and 2d cases.
    dmydx = gradm[1, 0]
    dmzdx = gradm[2, 0]

    # Derivatives along z do not exist in the 1d/2d cases; set to zero.
    dmydz = 0

    if dim == 1:
        # Derivatives along y do not exist in the 1d case; set to zero.
        dmxdy = 0
        dmzdy = 0
    elif dim == 2:
        dmxdy = gradm[0, 1]
        dmzdy = gradm[2, 1]
    else:
        raise ValueError("times_curl only supports dim in (1, 2, 3)")

    # Components of curl(m).
    curlx = dmzdy - dmydz
    curly = -dmzdx
    curlz = dmydx - dmxdy

    return m[0] * curlx + m[1] * curly + m[2] * curlz


def _dmi_interfacial(m, dim):
    """Return the interfacial-DMI bracket, transcribed from the legacy
    ``finmag.energies.dmi.DMI_interfacial``::

        (mx * dmzdx - mz * dmxdx) + (my * dmzdy - mz * dmydy)

    References: Rohart, S. and Thiaville A., Phys. Rev. B 88, 184422 (2013).
    """
    gradm = grad(m)

    dmxdx = gradm[0, 0]
    dmydx = gradm[1, 0]
    dmzdx = gradm[2, 0]

    if dim == 1:
        dmxdy = 0
        dmydy = 0
        dmzdy = 0
    else:
        # Works for both 2d and 3d meshes.
        dmxdy = gradm[0, 1]
        dmydy = gradm[1, 1]
        dmzdy = gradm[2, 1]

    mx = m[0]
    my = m[1]
    mz = m[2]

    return (mx * dmzdx - mz * dmxdx) + (my * dmzdy - mz * dmydy)


def DMI_interfacial(m, D, dim):
    """Return the interfacial-DMI energy-density UFL form, preserving the
    legacy public signature (``m`` a :class:`~finmag.field.Field`, ``D`` an
    already-scaled coefficient expression, ``dim`` the mesh dimension)."""
    return D * _dmi_interfacial(m.f, dim)
