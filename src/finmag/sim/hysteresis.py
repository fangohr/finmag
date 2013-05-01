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

def hysteresis(sim, H_ext_list, fun=None, save_every=None,
               save_at_stage_end=True, **kwargs):
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

        H_ext_list:  list of 3-vectors

            List of external fields, where each field can have any of
            the forms accepted by Zeeman.__init__() (see its docstring
            for more details).

        fun:  callable

            The user can pass a function here (which should accept the
            Simulation object as its only argument); this function is
            called after each relaxation and can be used for example
            to save the value of the averaged magnetisation.

        save_every:  float | None

            Time interval between subsequent vtk snapshots during the
            simulation (the default is `None`, which means not to save
            regular vtk snapshots).

        save_at_stage_end:  bool

            Whether a vtk snapshot of the final relaxed state should
            be saved after each stage (default: True).

    For a list of other keyword arguments accepted by this method see
    the documentation of the relax() method, to which all given keyword
    arguments are passed on. Note that if a `filename` argument is
    provided, the string 'stage_xxx' is appended to it, where xxx is
    a running counter which indicates for which field in H_ext_list
    the relax() method is being executed.

    *Return value*

    If `fun` is not None then the return value is a list containing an
    accumulation of all the return values of `fun` after each stage.
    Otherwise the return value is None.

    """
    if H_ext_list == []:
        return

    save_vtk_snapshots = (save_every is not None or save_at_stage_end == True)
    filename = kwargs.get('filename', None)

    if filename is None:
        if save_vtk_snapshots == True:
            log.warning("The keywords 'save_every' and 'save_at_stage_end' "
                        "will be ignored because no filename was given. "
                        "Please provide one if you would like to save VTK "
                        "snapshots.")
        save_every = None
        save_at_stage_end = False
        save_vtk_snapshots = False

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

            if save_vtk_snapshots:
                if filename != '':
                    cur_filename = filename + "__stage_{:03d}__.pvd".format(cur_stage)
                item = sim.schedule('save_vtk', every=save_every,
                                    at_end=save_at_stage_end,
                                    filename=cur_filename)
            # XXX TODO: After the recent run_until refactoring the
            # relax() method doesn't accept a filename any more. We
            # need to schedule the snapshot saving ourselves here, or
            # ask the user to do it! (Need to think which alternative
            # is better.) -- Max, 30.1.2013
            sim.relax(**kwargs)
            if save_vtk_snapshots:
                sim.unschedule(item)
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

    if save_vtk_snapshots:
        # Here we create two 'global' output files: one which combines
        # all .vtu files produced during the simulation, and one which
        # comprises only the final relaxed states of each stage.

        # Remove trailing underscores from output .pvd files
        # (for cosmetic reasons only... ;-)).
        for f in sorted(glob.glob(filename + "__stage_[0-9][0-9][0-9]__.pvd")):
            os.rename(f, re.sub('__\.pvd', '.pvd', f))

        if save_every is not None:
            # First we write the file containing *all* .vtu files.
            f_global = open(filename + '_all.pvd', 'w')
            f_global.write(textwrap.dedent("""\
                <?xml version="1.0"?>
                <VTKFile type="Collection" version="0.1">
                  <Collection>
                """))

            cur_stage = 0
            cur_timestep = 0
            for f in sorted(glob.glob(filename + "__stage_[0-9][0-9][0-9].pvd")):
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
                cur_stage += 1
            f_global.write("  </Collection>\n</VTKFile>\n")
            f_global.close()

        if save_at_stage_end == True:
            # Write the last .vtu file produced during each stage to a
            # global .pvd file.
            f_global = open(filename + '.pvd', 'w')
            f_global.write(textwrap.dedent("""\
                <?xml version="1.0"?>
                <VTKFile type="Collection" version="0.1">
                  <Collection>
                """))

            for i in xrange(num_stages):
                f_global.write("    <!-- Hysteresis stage #{:03d} -->\n".format(i))
                vtu_files = sorted(glob.glob(filename + '__stage_{:03d}__*.vtu'.format(i)))
                f_global.write('    <DataSet timestep="{}" part="0" file="{}" />\n'.format(i, os.path.basename(vtu_files[-1])))
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
