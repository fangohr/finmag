def run_with_schedule(integrator, schedule, callbacks_at_scheduler_events=[]):
    """
    Integrate time until an exit condition in the schedule has been met.

    The optional argument `callbacks_at_scheduler_events` should be a
    list of functions which are called whenever the time integration
    reaches a "checkpoint" where some event is scheduled. Each such
    function should expect the timestep t at which the event occurs as
    its single argument. Note that these functions are called just
    *before* the scheduled events are triggered. This is used, for
    example, to keep time-dependent fields up to date with the
    simulation time.

    """
    schedule.start_realtime_jobs()

    for t in schedule:
        assert(t >= integrator.cur_t)  # sanity check

        # If new items were scheduled after a previous time
        # integration finished, we can have t == integrator.cur_t.
        # However, this confuses the integrators so we don't integrate
        # in this case.
        if t != integrator.cur_t:
            integrator.advance_time(t)

        for f in callbacks_at_scheduler_events:
            f(t)
        schedule.reached(t)

    schedule.finalise(t)
    schedule.stop_realtime_jobs()
