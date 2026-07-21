"""Backend-selecting integrator factory (unchanged public API).

Task 8 (superseded by Task 20, see below): the native Sundials/CVODE
extension was not yet ported to DOLFINx, so ``backend="sundials"`` kept
raising ``ImportError`` by name. Instead of adding a new selection mechanism,
the ``backend`` keyword's *default value* was changed from ``"sundials"`` to
``"scipy"``, making the ported ``ScipyIntegrator`` the temporary
supported/default DOLFINx backend while every legacy caller that still
explicitly passes ``backend="sundials"`` (e.g. ``finmag.sim.sim.Simulation``)
was unaffected.

Task 20 fix round 1: the native Sundials/CVODE backend is now fully ported
and validated on DOLFINx (see ``sundials_integrator.py`` and the
``dolfinx-src-sundials-pytest`` gate). With the backend genuinely available,
this factory's ``backend`` default is flipped back to ``"sundials"``,
restoring the legacy default semantics, exactly as the Task 20 plan directed.
This is low blast radius: ``finmag.sim.sim.Simulation`` passes
``backend=self.integrator_backend`` explicitly (``sim.py`` around
``create_integrator``/``reset_time``) and every other in-tree caller passes
``backend`` explicitly. When the native extension is unavailable in an
environment, a bare ``llg_integrator(llg, m0)`` call now raises ``ImportError``
by name (unchanged failure mode, just triggered by the default instead of an
explicit request) -- callers that need the always-available driver should
still pass ``backend="scipy"`` explicitly.

Note ``Simulation.integrator_backend`` itself is a *separate* default (set in
``finmag.sim.sim.Simulation.__init__``) and deliberately stays ``"scipy"`` for
now -- see the "USER ACCEPTANCE PENDING" register entry in
``transition-notes.org`` / ``dev/dolfinx/porting_map.md`` / the Task 20 plan
section. [Claude Sonnet 5]

Task 11a: the eager module-scope ``from finmag.drivers.sundials_integrator
import SundialsIntegrator`` was moved behind a PEP 562 module ``__getattr__``.
Importing ``sundials_integrator`` pulls in ``finmag.native``, which triggers a
native build; on the DOLFINx import path that would leave ``finmag.native`` in
``sys.modules`` and violate the "plain import loads no native modules"
boundary. Previously this only stayed clean by accident, because the native
build *failed* in the DOLFINx env. Now that an array-only native surface can
build there, the sundials availability probe is loaded lazily: the module
attributes ``SundialsIntegrator`` / ``SUNDIALS_INTEGRATOR_IMPORT_ERROR`` still
exist and reflect real availability on first access, but plain ``import
finmag`` / ``sim_with`` never touches ``finmag.native``. [Claude Opus 4.8]
"""

import logging
from finmag.field import Field
try:
    from finmag.drivers.scipy_integrator import ScipyIntegrator
    SCIPY_INTEGRATOR_IMPORT_ERROR = None
except Exception as error:
    ScipyIntegrator = None
    SCIPY_INTEGRATOR_IMPORT_ERROR = error

log = logging.getLogger(name='finmag')

_SUNDIALS_PROBED = False


def _probe_sundials_integrator():
    """Lazily import the native Sundials integrator, caching the outcome.

    Returns the ``SundialsIntegrator`` class, or ``None`` when the native
    extension is unavailable in this environment. The matching import error is
    cached in the module global ``SUNDIALS_INTEGRATOR_IMPORT_ERROR``. This is
    deferred (not run at module import) so that merely importing this module --
    which happens on the plain ``import finmag`` / ``sim_with`` path -- never
    imports ``finmag.native``. [Claude Opus 4.8]
    """
    global _SUNDIALS_PROBED, SundialsIntegrator, SUNDIALS_INTEGRATOR_IMPORT_ERROR
    if not _SUNDIALS_PROBED:
        import sys
        native_before = {
            name for name in sys.modules
            if name == "finmag.native" or name.startswith("finmag.native.")
        }
        try:
            from finmag.drivers.sundials_integrator import (
                SundialsIntegrator as _SundialsIntegrator,
            )
            SundialsIntegrator = _SundialsIntegrator
            SUNDIALS_INTEGRATOR_IMPORT_ERROR = None
        except Exception as error:
            SundialsIntegrator = None
            SUNDIALS_INTEGRATOR_IMPORT_ERROR = error
            # Importing sundials_integrator pulls `from finmag.native import
            # sundials`. In the DOLFINx env the finmag.native *package* now
            # imports successfully (its __init__ can build the array-only
            # `bem_arrays` surface) before the unported `sundials` submodule
            # fails. Roll back that residue so a failed sundials probe leaves no
            # finmag.native modules loaded, preserving the ported scipy/core
            # "no native on the import path" boundary the DOLFINx tests assert.
            # (On the legacy stack the import succeeds and nothing is removed.)
            # [Claude Opus 4.8]
            for name in list(sys.modules):
                if (name == "finmag.native" or name.startswith("finmag.native.")) \
                        and name not in native_before:
                    del sys.modules[name]
        _SUNDIALS_PROBED = True
    return SundialsIntegrator


def __getattr__(name):
    # PEP 562: resolve the deferred sundials availability symbols on first
    # access, so `from finmag.drivers.llg_integrator import SundialsIntegrator`
    # (and SUNDIALS_INTEGRATOR_IMPORT_ERROR) keep working and reflect real
    # availability, without importing finmag.native at module load. Once probed,
    # the names become real module globals and this hook is not consulted again.
    if name in ("SundialsIntegrator", "SUNDIALS_INTEGRATOR_IMPORT_ERROR"):
        _probe_sundials_integrator()
        return globals()[name]
    raise AttributeError(
        "module {!r} has no attribute {!r}".format(__name__, name)
    )


def llg_integrator(llg, m0, backend="sundials", **kwargs):
    # XXX TODO: Passing the tablewriter argument on like this is a
    #           complete hack and this should be refactored. The same
    #           is true with saving snapshots. Neither saving average
    #           fields nor VTK snapshots should probably happen in
    #           this class but rather in the Simulation class (?).
    #             -- Max, 11.12.2012
    #           Yes, I think that's right. We could give callback functions
    #           to the run_until and relax function to give control back to the
    #           simulation class.
    #             -- Hans, 17/12/2012
    #
    assert isinstance(m0, Field)

    log.info("Creating integrator with backend {} and arguments {}.".format(backend, kwargs))
    if backend == "scipy":
        if ScipyIntegrator is None:
            raise ImportError(
                "The 'scipy' integrator backend is not available in this "
                "environment: {}".format(SCIPY_INTEGRATOR_IMPORT_ERROR)
            )
        return ScipyIntegrator(llg, m0, **kwargs)
    elif backend == "sundials":
        integrator_cls = _probe_sundials_integrator()
        if integrator_cls is None:
            # The pixi probe allows SciPy-backed stepping before the native
            # Sundials extension is ported to the newer conda-forge stack. [Codex GPT-5.4]
            raise ImportError(
                "The 'sundials' integrator backend is not available in this "
                "environment: {}".format(SUNDIALS_INTEGRATOR_IMPORT_ERROR)
            )
        return integrator_cls(llg, m0.get_ordered_numpy_array_xxx(), **kwargs)
    else:
        raise ValueError("backend must be either scipy or sundials")
