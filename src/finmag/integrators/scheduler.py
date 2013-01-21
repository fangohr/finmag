import atexit
import logging
import functools
from numbers import Number
from datetime import datetime, timedelta
# This module will try to import the package apscheduler when a realtime event
# is added. Install with "pip install apscheduler".
# See http://pypi.python.org/pypi/APScheduler for documentation.

log = logging.getLogger(name="finmag")

EPSILON = 1e-15 # femtosecond precision for scheduling.

class At(object):
    """
    Store a function and a time when that function should be run.
    Call that function if notified that the timing is right.

    """
    def __init__(self, time, at_end=False):
        """
        Initialise with the correct time.

        """
        self.last_step = None
        self.next_step = time
        self.callback = None
        self.at_end = at_end

    def call(self, callback):
        """
        Attach a callback.

        """
        self.callback = callback

        # so that object can be initialised and a callback attached
        # in one line:  at = At(5.0e-12).call(my_fun)
        return self

    attach = call

    def fire(self, time):
        """
        Call registered function.

        """
        if (self.last_step != None) and abs(self.last_step - time) < EPSILON:
            # Don't fire more than once per time. This would be possible if the
            # scheduled time also happens to be the end of the simulation and
            # at_end was set to True.
            return

        if self.callback:
            self.callback()
        self.last_step = time
        self.update()

    def update(self):
        """
        Compute next target time.

        """
        self.next_step = None # Since one-time event, there is no next step.


class Every(At):
    """
    Store a function that should be run and a time interval between function calls.
    Call that function if notified that the timing is right.

    """
    def __init__(self, interval, start=None, at_end=False):
        """
        Initialise with the interval between correct times and optionally, a starting time.

        """
        self.last_step = None
        self.next_step = start or 0.0
        self.interval = interval
        self.callback = None
        self.at_end = at_end

    def update(self):
        """
        Compute next target time.

        """
        self.next_step += self.interval


class Scheduler(object):
    """
    Manages a list of actions that should be performed at specific times.

    """
    def __init__(self):
        """
        Create a Scheduler.

        """
        self.items = []

    def add(self, func, args=None, kwargs=None, at=None, at_end=False, every=None, after=None, realtime=False):
        """
        Register a function with the scheduler.

        """
        if not hasattr(func, "__call__"):
            raise TypeError("The function must be callable but object '%s' is of type '%s'" % \
                (str(func), type(func)))
        assert at or every or at_end or (after and realtime), "Use either `at`, `every` or `at_end` if not in real time mode."
        assert not (at!=None and every!=None), "It's either `at` or `every`."
        assert not (at!=None and after!=None), "Delays don't mix with `at`."

        args = args or []
        kwargs = kwargs or {}
        callback = functools.partial(func, *args, **kwargs)

        if realtime:
            self._add_realtime(func, at, every, after)
            if at_end:
                at_end_item = At(None, True).call(callback)
                self._add(at_end_item)
            return

        if at or (at_end and not every):
            at_item = At(at, at_end).call(callback)
            self._add(at_item)
            return

        if every:
            every_item = Every(every, after, at_end).call(callback)
            self._add(every_item)

    def _add(self, at_or_every):
        self.items.append(at_or_every)

    def _add_realtime(self, func, at=None, every=None, after=None):
        if not hasattr(self, "apscheduler"):
            try:
                from apscheduler.scheduler import Scheduler as APScheduler
            except ImportError:
                log.error("Need APScheduler package to schedule realtime events.\n"
                    "Please install from http://pypi.python.org/pypi/APScheduler.")
                raise

            self.apscheduler = APScheduler()
            atexit.register(lambda: self.apscheduler.shutdown(wait=False))
            self.apscheduler.start()

        if after and isinstance(after, Number):
            # `after` can be either a delay in seconds, or a date/datetime.
            # Since the APScheduler API expects a date/datetime convert it. 
            after = datetime.now() + timedelta(seconds=after)

        if at:
            self.apscheduler.add_date_job(func, at)
        elif every:
            if after:
                self.apscheduler.add_interval_job(func, seconds=every, start_date=after)
            else:
                self.apscheduler.add_interval_job(func, seconds=every)
        elif after:
            self.apscheduler.add_date_job(func, after)
        else:
            raise ValueError("Assertion violated. Use either `at`, `every` of `after`.")

    def next_step(self):
        """
        Returns the time for the next action to be performed.

        """
        next_steps = [i.next_step for i in self.items if i.next_step is not None]
        if len(next_steps) == 0:
            return None
        return min(next_steps)

    def reached(self, time):
        """
        Notify the Scheduler that a certain point in time has been reached.

        It will perform the action(s) that were defined before.

        """
        for item in self.items:
            if (item.next_step != None) and abs(item.next_step - time) < EPSILON:
                item.fire(time)

    def finalise(self, time):
        """
        Trigger all events that need to happen at the end of time integration.

        """
        for item in self.items:
            if item.at_end:
                item.fire(time)

def save_ndt(sim):
    """Given the simulation object, saves one line to the ndt finalise
    (through the TableWriter object)."""
    if sim.driver == 'cvode':
        log.debug("Saving data to ndt file at t={} (sim.name={}).".format(sim.t, sim.name))
    else:
        raise NotImplementedError("Only cvode driver known.")
    sim.tablewriter.save()

