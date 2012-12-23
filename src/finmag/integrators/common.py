import time
import logging
import numpy as np
import dolfin as df

log = logging.getLogger(name="finmag")

ONE_DEGREE_PER_NS = 17453292.5  # in rad/s

def run_until_relaxation(integrator,
        save_snapshots=False, filename=None, save_every=100e-12, save_final_snapshot=True,
        stopping_dmdt=ONE_DEGREE_PER_NS, dmdt_increased_counter_limit=50, dt_limit=1e-10):
    """
    Run integration until the maximum |dm/dt| is smaller than the
    threshold value stopping_dmdt (which is one degree per
    nanosecond per default).

    As a precaution against running an infinite amount of time when
    |dm/dt| - stopping_dmdt doesn't convergence (because of badly
    chosen tolerances?), the integration will stop if |dm/dt|
    increases `dmdt_increased_counter_limit` times during the integration
    (default value: 50). The maximum allowed timestep per integration step
    can be controlled via `dt_limit`.

    If save_snapshots is True (default: False) then a series of snapshots
    is saved to `filename` (which must be specified in this case). If
    `filename` contains directory components then these are created if they
    do not already exist. A snapshot is saved every `save_every` seconds
    (default: 100e-12, i.e. every 100 picoseconds). It should be noted that
    the true timestep at which the snapshot is saved may deviate from slightly
    from the exact value due to the way the time integrators work.
    Usually, one last snapshot is saved after the relaxation is finished (or
    was stopped). This can be disabled by setting save_final_snapshot to False
    (default: True).

    """
    dt = 1e-14 # initial timestep (TODO: use the characteristic time here)

    dt_increment_multi = 1.5;
    dmdt_increased_counter = 0;

    if save_snapshots:
        f = df.File(filename, 'compressed')

    cur_count = 0  # current snapshot count
    start_time = integrator.cur_t  # start time of the integration; needed for snapshot saving
    last_max_dmdt_norm = 1e99
    while True:
        prev_m = integrator.llg.m.copy()
        next_stop = integrator.cur_t + dt

        # If in the next step we would cross a timestep where a snapshot should be saved, run until
        # that timestep, save the snapshot, and then continue.
        while save_snapshots and (next_stop >= start_time+cur_count*save_every):
            integrator.run_until(cur_count*save_every)
            integrator._do_save_snapshot(f, cur_count, filename, save_averages=True)
            cur_count += 1

        integrator.run_until(next_stop)

        dm = np.abs(integrator.m - prev_m).reshape((3, -1))
        dm_norm = np.sqrt(dm[0] ** 2 + dm[1] ** 2 + dm[2] ** 2)
        max_dmdt_norm = float(np.max(dm_norm) / dt)

        if max_dmdt_norm < stopping_dmdt:
            log.debug("{}: Stopping at t={:.3g}, with last_dmdt={:.3g}, smaller than stopping_dmdt={:.3g}.".format(
                integrator.__class__.__name__, integrator.cur_t, max_dmdt_norm, float(stopping_dmdt)))
            break

        if dt < dt_limit / dt_increment_multi:
            if not max_dmdt_norm > last_max_dmdt_norm:
                dt *= dt_increment_multi
        else:
            dt = dt_limit

        log.debug("{}: t={:.3g}, last_dmdt={:.3g} * stopping_dmdt, next dt={:.3g}.".format(
            integrator.__class__.__name__, integrator.cur_t, max_dmdt_norm / stopping_dmdt, dt))

        if max_dmdt_norm > last_max_dmdt_norm:
            dmdt_increased_counter += 1
            log.debug("{}: dmdt {:.2f} times larger than last time (counting {}/{}).".format(
                integrator.__class__.__name__, max_dmdt_norm / last_max_dmdt_norm,
                dmdt_increased_counter, dmdt_increased_counter_limit))

        last_max_dmdt_norm = max_dmdt_norm

        if dmdt_increased_counter >= dmdt_increased_counter_limit:
            log.warning("{}: Stopping after it increased {} times.".format(
                integrator.__class__.__name__, dmdt_increased_counter_limit))
            break

    if save_snapshots and save_final_snapshot:
        _do_save_snapshot(integrator, f, cur_count, filename, save_averages=True)

def _do_save_snapshot(integrator, f, cur_count, filename, save_averages=True):
    # TODO: Can we somehow store information about the current timestep in either the .pvd/.vtu file itself, or in the filenames?
    #       Unfortunately, it seems as if the filenames of the *.vtu files are generated automatically.
    t0 = time.time()
    f << integrator.llg._m
    t1 = time.time()
    log.debug("Saving snapshot #{} at timestep t={:.4g} to file '{}' (saving took {:.3g} seconds).".format(
        cur_count, integrator.cur_t, filename, t1 - t0))
    if save_averages:
        if integrator.tablewriter:
            log.debug("Saving average field values (in integrator).")
            integrator.tablewriter.save()
        else:
            log.warning("Cannot save average fields because no Tablewriter is present in integrator.")
