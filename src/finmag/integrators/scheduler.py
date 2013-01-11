import atexit
import logging
from numbers import Number
from datetime import datetime, timedelta
# This module will try to import the package apscheduler when a realtime event
# is added. Install with "pip install apscheduler".
# See http://pypi.python.org/pypi/APScheduler for documentation.

log = logging.getLogger(name="finmag")

class At(object):
    """
    Store a function and a time when that function should be run.
    Call that function if notified that the timing is right.

    """
    def __init__(self, time, at_end=False):
        """
        Initialise with the correct time.

        """
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

    def fire(self, payload=None):
        """
        Call registered function with an optional `payload` as argument.

        """
        if self.callback:
            self.callback(payload) if payload else self.callback()
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
    def __init__(self, payload=None):
        """
        Creates a Scheduler with an optional payload. 
        If there is a payload, registered functions will be called with it as an argument.

        """
        self.items = []
        self.payload = payload

    def add(self, func, at=None, at_end=False, every=None, after=None, realtime=False):
        """
        Register a function with the scheduler.

        """
        assert at or every or at_end or (after and realtime), "Use either `at`, `every` or `at_end` if not in real time mode."
        assert not (at!=None and every!=None), "It's either `at` or `every`."
        assert not (at!=None and after!=None), "Delays don't mix with `at`."

        if realtime:
            self._add_realtime(func, at, every, after)
            if at_end:
                at_end_item = At(None, True).call(func)
                self._add(at_end_item)
            return

        if at or (at_end and not every):
            at_item = At(at, at_end).call(func)
            self._add(at_item)
            return

        if every:
            every_item = Every(every, after, at_end).call(func)
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
            if item.next_step == time:
                item.fire(self.payload)

    def finalise(self):
        """
        Trigger all events that need to happen at the end of time integration.

        """
        for item in self.items:
            if item.at_end:
                item.fire(self.payload)
