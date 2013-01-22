import logging
import numpy as np

log = logging.getLogger(name="finmag")

ONE_DEGREE_PER_NS = 17453292.5  # in rad/s

def run_with_schedule(integrator, time, schedule=None):
    """
    Run the integrator until *time* has been reached.

    When a scheduler has been passed as an argument, the integrator will stop
    at the defined time-steps and notify the scheduler so it can trigger the
    appropriate actions.

    IDEA:
       let time also be 'relaxation' or something similar to unify both methods
       if possible (if we run until relaxation, we will need to observe
       convergence and have a default schedule if none is provided.

    """
    if not schedule:
        # If no schedule is passed the integrator can run until the final
        # time and stop. This replicates the initial behaviour of run_until.
        return integrator.advance_time(time)

    schedule.start_realtime_jobs()

    while True:
        next_step = schedule.next_step()
        if next_step == None or next_step > time:
            break

        integrator.advance_time(next_step)
        schedule.reached(next_step)

    if integrator.cur_t < time:
        integrator.advance_time(time)

    schedule.finalise(time)
    schedule.stop_realtime_jobs()


def relax_with_schedule(integrator,
        stopping_dmdt=ONE_DEGREE_PER_NS, dmdt_increased_counter_limit=500, dt_limit=1e-10,
        schedule=None):
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

    """
    if schedule:
        schedule.start_realtime_jobs()

    dt = 1e-14 # initial timestep (TODO: use the characteristic time here)

    dt_increment_multi = 1.5;
    dmdt_increased_counter = 0;

    last_max_dmdt_norm = 1e99
    while True:
        prev_m = integrator.llg.m.copy()
        next_stop = integrator.cur_t + dt

        if schedule:
            next_stop_on_schedule = schedule.next_step()
            while next_stop_on_schedule != None and next_stop_on_schedule < next_stop:
                integrator.advance_time(next_stop_on_schedule)
                schedule.reached(next_stop_on_schedule)
                next_stop_on_schedule = schedule.next_step()

        integrator.advance_time(next_stop)
        if schedule:
            schedule.reached(next_stop)

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
            #log.debug("{}: dmdt {:.2f} times larger than last time (counting {}/{}).".format(
            log.debug("{}: dmdt {} times larger than last time (counting {}/{}).".format(
                integrator.__class__.__name__, max_dmdt_norm / last_max_dmdt_norm,
                dmdt_increased_counter, dmdt_increased_counter_limit))

        last_max_dmdt_norm = max_dmdt_norm

        if dmdt_increased_counter >= dmdt_increased_counter_limit:
            log.warning("{}: Stopping after dmdt increased {} times.".format(
                integrator.__class__.__name__, dmdt_increased_counter_limit))
            break

    if schedule:
        schedule.finalise(integrator.cur_t)
        schedule.stop_realtime_jobs()
