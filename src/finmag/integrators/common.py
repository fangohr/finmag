def run_with_schedule(integrator, schedule):
    """
    Integrate time until an exit condition in the schedule has been met.

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

        schedule.reached(t)

    schedule.finalise(t)
    schedule.stop_realtime_jobs()


