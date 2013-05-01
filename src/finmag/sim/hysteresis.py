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

def hysteresis(sim, H_ext_list, fun=None, save_snapshots=False, **kwargs):
    """
    Set the applied field to the first value in `H_ext_list` (which should
    be a list of external field vectors) and then call the relax() method.
    When convergence is reached, the field is changed to the next one in
    H_ext_list, and so on until all values in H_ext_list are exhausted.

    Note: The fields in H_ext_list are applied *in addition to* any Zeeman
          interactions that are already present in the simulation.
          In particular, if only one external field should be present then
          do not add any Zeeman interactions before calling this method.

    *Arguments*

        H_ext_list -- list of external fields, where each field can have
                      any of the forms accepted by Zeeman.__init__()
                      (see its docstring for more details)

       fun -- the user can pass a function here (which should accept the
              Simulation object as its only argument); this function is
              called after each relaxation and can be used for example to
              save the value of the averaged magnetisation.

    If `fun` is not None then this function returns a list containing
    an accumulation of all the return values of `fun` after each stage.
    Otherwise this function returns None.

    For a list of keyword arguments accepted by this method see the
    documentation of the relax() method, to which all given keyword
    arguments are passed on. Note that if a `filename` argument is
    provided, the string 'stage_xxx' is appended to it, where xxx is
    a running counter which indicates for which field in H_ext_list
    the relax() method is being executed.

    """
    if H_ext_list == []:
        return

    filename = kwargs.get('filename', None)
    force_overwrite = kwargs.get('force_overwrite', False)
    if filename != None and force_overwrite == True:
        if os.path.exists(filename):
            # Delete the global .pvd file as well as all existing .pvd
            # and .vtu file from any previously run hysteresis stages.
            # Although the relax() command also checks for existing files,
            # it would miss any stages that were previously run but are
            # not reached during this run.
            log.debug("Removing the file '{}' as well as all associated "
                      ".pvd and .vtu files of previously run hysteresis "
                      "stages.".format(filename))
            pvdfiles = glob.glob(re.sub('\.pvd$', '', filename) + '*.pvd')
            vtufiles = glob.glob(re.sub('\.pvd$', '', filename) + '*.vtu')
            for f in pvdfiles + vtufiles:
                os.remove(f)

    # Add a new Zeeman interaction, initialised to zero.
    H = Zeeman((0, 0, 0))
    sim.add(H)

    # We keep track of the current stage of the hysteresis loop.
    # Each stage is saved to a different .pvd file, whose name
    # includes the current stage number.
    cur_stage = 0
    num_stages = len(H_ext_list)
    filename = re.sub('\.pvd$', '', kwargs.pop('filename', ''))
    cur_filename = ''

    res = []

    try:
        while True:
            H_cur = H_ext_list[cur_stage]
            log.info("Entering hysteresis stage #{} "
                     "({} out of {}).".format(cur_stage,
                                              cur_stage + 1, num_stages))
            H.set_value(H_cur)

            if filename != '':
                cur_filename = filename + "__stage_{:03d}__.pvd".format(cur_stage)
            # XXX TODO: After the recent run_until refactoring the
            # relax() method doesn't accept a filename any more. We
            # need to schedule the snapshot saving ourselves here, or
            # ask the user to do it! (Need to think which alternative
            # is better.) -- Max, 30.1.2013
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
    sim.llg.effective_field.interactions.remove(H)

    if save_snapshots:
        # We now remove trailing underscores from output filenames
        # (for cosmetic resons only ... ;-) and create a 'global'
        # output file which combines all stages of the simulation.
        #
        f_global = open(filename + '.pvd', 'w')
        f_global.write(textwrap.dedent("""\
            <?xml version="1.0"?>
            <VTKFile type="Collection" version="0.1">
              <Collection>
            """))

        cur_stage = 0
        cur_timestep = 0
        for f in sorted(glob.glob(filename + "__stage_[0-9][0-9][0-9]__.pvd")):
            f_global.write("    <!-- Hysteresis stage #{:03d} -->\n".format(cur_stage))
            for line in fileinput.input([f]):
                if re.match('^\s*<DataSet .*/>$', line):
                    # We require sequentially increasing timesteps, so
                    # we have to manually substitue them (TODO: unless
                    # we can already do this when saving the snapshot?!?)
                    line = re.sub('timestep="[0-9]+"', 'timestep="{}"'.format(cur_timestep), line)
                    f_global.write(line)
                    cur_timestep += 1
            f_global.write("\n")
            os.rename(f, re.sub('__\.pvd', '.pvd', f))
            cur_stage += 1
        f_global.write("  </Collection>\n</VTKFile>\n")
        f_global.close()

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
