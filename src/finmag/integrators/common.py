def run_with_schedule(integrator, schedule):
    """
    Integrate time until an exit condition in the schedule has been met.

    """
    schedule.start_realtime_jobs()

    for t in schedule:
        integrator.advance_time(t)
        schedule.reached(t)

    schedule.finalise(t)
    schedule.stop_realtime_jobs()
