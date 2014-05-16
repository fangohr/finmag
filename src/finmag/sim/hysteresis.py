import os
import re
import glob
import logging
import textwrap
import fileinput
import numpy as np
from finmag.energies import Zeeman
from finmag.util.helpers import norm

log = logging.getLogger(name="finmag")

def hysteresis(sim, H_ext_list, fun=None, **kwargs):
    """
    Set the applied field to the first value in `H_ext_list` (which should
    be a list of external field vectors) and then call the relax() method.
    When convergence is reached, the field is changed to the next one in
    H_ext_list, and so on until all values in H_ext_list are exhausted.

    Note: The fields in H_ext_list are applied *in addition to* any Zeeman
          interactions that are already present in the simulation.
          In particular, if only one external field should be present then
          do not add any Zeeman interactions before calling this method.

    If you would like to perform a certain action (e.g. save a VTK
    snapshot of the magnetisation) at the end of each relaxation stage,
    use the sim.schedule() command with the directive 'at_end=True' as
    in the following example:

        sim.schedule('save_vtk', at_end=True, ...)
        sim.hysteresis(...)


    *Arguments*

        H_ext_list:  list of 3-vectors

            List of external fields, where each field can have any of
            the forms accepted by Zeeman.__init__() (see its docstring
            for more details).

        fun:  callable

            The user can pass a function here (which should accept the
            Simulation object as its only argument); this function is
            called after each relaxation and determines the return
            value (see below). For example, if

               fun = (lambda sim: sim.m_average[0])

            then the return value is a list of values representing the
            average x-component of the magnetisation at the end of
            each relaxation.

    All other keyword arguments are passed on to the relax() method.
    See its documentation for details.


    *Return value*

    If `fun` is not None then the return value is a list containing an
    accumulation of all the return values of `fun` after each stage.
    Otherwise the return value is None.

    """
    if H_ext_list == []:
        return

    # Add a new Zeeman interaction, initialised to zero.
    H = Zeeman((0, 0, 0))
    sim.add(H)

    # We keep track of the current stage of the hysteresis loop.
    cur_stage = 0
    num_stages = len(H_ext_list)

    res = []

    try:
        while True:
            H_cur = H_ext_list[cur_stage]
            log.info(
                "Entering hysteresis stage #{} ({} out of {}). Current field: "
                "{}".format(cur_stage, cur_stage + 1, num_stages, H_cur))
            H.set_value(H_cur)
            sim.relax(**kwargs)
            cur_stage += 1
            if fun is not None:
                retval = fun(sim)
                res.append(retval)
                log.debug("hysteresis callback function '{}' returned "
                          "value: {}".format(fun.__name__, retval))
    except IndexError:
        log.info("Hysteresis is finished.")

    log.info("Removing the applied field used for hysteresis.")
    sim.remove_interaction(H.name)

    return res or None


def hysteresis_loop(sim, H_max, direction, N, **kwargs):
    """
    Compute a hysteresis loop. This is a specialised convenience
    version of the more general `hysteresis` method. It computes a
    hysteresis loop where the external field is applied along a
    single axis and changes magnitude from +H_max to -H_max and
    back (using N steps in each direction).

    The return value is a pair (H_vals, m_vals), where H_vals is
    the list of field strengths at which a relaxation is performed
    and m_vals is a list of scalar values containing, for each
    field value, the averaged value of the magnetisation along the
    axis `direction` (after relaxation has been reached). Thus the
    command plot(H_vals, m_vals) could be used to plot the
    hysteresis loop.

       direction -- a vector indicating the direction of the
                    external field (will be normalised
                    automatically)

       H_max -- maximum field strength

       N -- number of data points to compute in each direction
            (thus the total number of data points for the entire
            loop will be 2*N-1)

       kwargs -- any keyword argument accepted by the hysteresis() method
    """
    d = np.array(direction)
    H_dir = d / norm(d)
    H_norms = list(np.linspace(H_max, -H_max, N)) + \
        list(np.linspace(-H_max, H_max, N))
    H_vals = [h * H_dir for h in H_norms]
    m_avg = hysteresis(sim, H_vals, fun=lambda sim: sim.m_average, **kwargs)
    # projected lengths of the averaged magnetisation values along the axis `H_dir`
    m_vals = [np.dot(m, H_dir) for m in m_avg]
    return (H_norms, m_vals)
