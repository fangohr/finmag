"""Backend-selecting integrator factory (unchanged public API).

Task 8: the native Sundials/CVODE extension is not yet ported to DOLFINx, so
``backend="sundials"`` keeps raising ``ImportError`` by name (unchanged
behaviour, see the ``sundials`` branch below). Instead of adding a new
selection mechanism, the ``backend`` keyword's *default value* is changed
from ``"sundials"`` to ``"scipy"``, making the ported ``ScipyIntegrator`` the
temporary supported/default DOLFINx backend while every legacy caller that
still explicitly passes ``backend="sundials"`` (e.g. ``finmag.sim.sim.Simulation``)
is unaffected. [Claude Sonnet 5]
"""

import logging
from finmag.field import Field
try:
    from finmag.drivers.sundials_integrator import SundialsIntegrator
    SUNDIALS_INTEGRATOR_IMPORT_ERROR = None
except Exception as error:
    SundialsIntegrator = None
    SUNDIALS_INTEGRATOR_IMPORT_ERROR = error
try:
    from finmag.drivers.scipy_integrator import ScipyIntegrator
    SCIPY_INTEGRATOR_IMPORT_ERROR = None
except Exception as error:
    ScipyIntegrator = None
    SCIPY_INTEGRATOR_IMPORT_ERROR = error

log = logging.getLogger(name='finmag')


def llg_integrator(llg, m0, backend="scipy", **kwargs):
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
        if SundialsIntegrator is None:
            # The pixi probe allows SciPy-backed stepping before the native
            # Sundials extension is ported to the newer conda-forge stack. [Codex GPT-5.4]
            raise ImportError(
                "The 'sundials' integrator backend is not available in this "
                "environment: {}".format(SUNDIALS_INTEGRATOR_IMPORT_ERROR)
            )
        return SundialsIntegrator(llg, m0.get_ordered_numpy_array_xxx(), **kwargs)
    else:
        raise ValueError("backend must be either scipy or sundials")
