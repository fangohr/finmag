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
    Calls that function when notified that the time is right.

    """
    def __init__(self, time):
        """
        Initialise with the correct time.

        """
        self.next_step = time
        self.callback = None

    def call(self, callback):
        """
        Attach a callback.

        """
        self.callback = callback

        # so that object can be initialised and a callback attached
        # in one line:  at = At(5.0e-12).call(my_fun)
        return self

    attach = call

    def fire(self):
        """
        Call the saved function.

        """
        self.next_step = None
        if self.callback:
            self.callback()
   
class Every(At):
    def __init__(self, interval, start=None, when_stopping=False):
        """
        Initialise with the interval between correct times and optionally, a starting time.
        If `when_stopping` is True, the event will be triggered a last time when
        time integration stops, even if less than `interval` time has passed.

        """
        self.next_step = start or 0.0
        self.interval = interval
        self.callback = None
        self.when_stopping = when_stopping

    def fire(self):
        """
        Call the saved function.

        """
        self.next_step += self.interval
        if self.callback:
            self.callback()

class Scheduler(object):
    """
    Manages a list of actions that should be performed at specific times.

    """
    def __init__(self):
        """
        Creates a Scheduler.

        """
        self.items = []

    def add(self, func, at=None, every=None, after=None, realtime=False):
        """
        Register a function with the scheduler.

        """
        assert at or every or (after and realtime), "Use either `at` or `every` if not in real time mode."
        assert not (at!=None and every!=None), "It's either `at` or `every`."
        assert not (at!=None and after!=None), "Delays don't mix with `at`."

        if realtime:
            self._add_realtime(func, at, every, after)
            return

        if at:
            at_item = At(at).call(func)
            self._add(at_item)

        if every:
            every_item = Every(every, after).call(func)
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
                item.fire()

    def finalise(self):
        """
        Trigger all events that need to happen at the end of time integration.

        """
        for item in self.items:
            if isinstance(item, Every) and item.when_stopping:
                item.fire()
