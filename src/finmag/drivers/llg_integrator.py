import logging
from finmag.field import Field
from finmag.drivers.sundials_integrator import SundialsIntegrator
from finmag.drivers.scipy_integrator import ScipyIntegrator

log = logging.getLogger(name='finmag')


def llg_integrator(physics, m, backend="sundials", **kwargs):
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
    log.info("Creating integrator with backend {} and arguments {}.".format(backend, kwargs))
    assert isinstance(m, Field)

    if backend == "scipy":
        return ScipyIntegrator(physics.hooks_scipy, m, **kwargs)
    elif backend == "sundials":
        return SundialsIntegrator(physics.hooks_sundials, m, **kwargs)
    else:
        raise ValueError("backend must be either scipy or sundials")
