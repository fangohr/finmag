import atexit
import logging
import functools
from numbers import Number
from datetime import datetime, timedelta
from finmag.integrators.events import SingleEvent, RepeatingEvent, same
# This module will try to import the package apscheduler when a realtime event
# is added. Install with "pip install apscheduler".
# See http://pypi.python.org/pypi/APScheduler for documentation.

log = logging.getLogger(name="finmag")
   
class Scheduler(object):
    """
    Manages a list of actions that should be performed at specific times.

    """
    def __init__(self):
        """
        Create a Scheduler.

        """
        self.items = []
        self.realtime_items = {}
        self.realtime_jobs = []  # while the scheduler is running, the job
                                 # associated with each realtime_item will be
                                 # stored in this list (otherwise it is empty)
        self.last = None

    def __iter__(self):
        return self

    def add(self, func, args=None, kwargs=None, at=None, at_end=False, every=None, after=None, realtime=False):
        """
        Register a function with the scheduler.

        """
        if not hasattr(func, "__call__"):
            raise TypeError("The function must be callable but object '%s' is of type '%s'" % \
                (str(func), type(func)))
        assert at or every or at_end or (after and realtime), "Use either `at`, `every` or `at_end` if not in real time mode."
        assert not (at!=None and every!=None), "Cannot mix `at` with `every`. Please schedule separately."
        assert not (at!=None and after!=None), "Delays don't mix with `at`."

        args = args or []
        kwargs = kwargs or {}
        callback = functools.partial(func, *args, **kwargs)

        if realtime:
            if at_end:
                at_end_item = SingleEvent(None, True).call(callback)
                self._add(at_end_item)
            return

        if at or (at_end and not every):
            at_item = SingleEvent(at, at_end).call(callback)
            self._add(at_item)
            return

        if every:
            every_item = RepeatingEvent(every, after, at_end).call(callback)
            self._add(every_item)

    def _add(self, item):
        self.items.append(item)

    def _remove(self, item):
        self.items.remove(item)

    def _add_realtime(self, func, at=None, every=None, after=None):
        """
        Add a realtime job.

        Returns the Job object as obtained from APScheduler.add_job() etc.

        """
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

        # Register the job so that it can be started/stopped as needed.
        self.realtime_items[func] = (at, every, after)

    def start_realtime_jobs(self):
        for (func, (at, every, after)) in self.realtime_items.items():
            if at:
                job = self.apscheduler.add_date_job(func, at)
            elif every:
                if after:
                    job = self.apscheduler.add_interval_job(func, seconds=every, start_date=after)
                else:
                    job = self.apscheduler.add_interval_job(func, seconds=every)
            elif after:
                job = self.apscheduler.add_date_job(func, after)
            else:
                raise ValueError("Assertion violated. Use either `at`, `every` of `after`.")

            self.realtime_jobs.append(job)

    def stop_realtime_jobs(self):
        for job in self.realtime_jobs:
            self.apscheduler.unschedule_job(job)
        self.realtime_jobs = []

    def next(self):
        """
        Returns the time for the next action to be performed.

        """
        next_step = None

        for item in self.items:
            if item.requests_stop_integration == True:
                raise StopIteration
            if item.next != None and (next_step == None or next_step > item.next):
                next_step = item.next

        if next_step == None:
            raise StopIteration

        if next_step < self.last:
            log.error("Scheduler computed the next time step should be t = {:.2g} s, but the last one was already t = {:.2g} s.".format(next_step, self.last))
            raise ValueError("Scheduler is corrupted. Requested a time step in the past: dt = {:.2g}.".format(next_step-self.last))
        return next_step

    def reached(self, time):
        """
        Notify the Scheduler that a certain point in time has been reached.

        It will perform the action(s) that were defined to happen at that time.

        """
        for item in self.items:
            if same(item.next, time):
                item.trigger(time)
        self.last = time

    def finalise(self, time):
        """
        Trigger all events that need to happen at the end of time integration.

        """
        for item in self.items:
            if item.trigger_on_stop:
                item.trigger(time, is_stop=True)

    def reset(self, time):
        """
        Override schedule so that internal time is now `time` and modify scheduled items accordingly.

        """
        self.last = None
        for item in self.items:
            item.reset(time)

    def _print_realtime_item(self, item, func_print=log.info):
        (f, (at, every, after)) = item
        func_print("'{}': <at={}, every={}, after={}>".format(
                item.callback.f.__name__, at, every, after))

    def print_scheduled_items(self, func_print=log.info):
        for item in self.items:
            print item # this will call __str__ on the item, which should be defned for all events
        for item in self.realtime_items:
            self._print_realtime_item(item, func_print)

    def clear(self):
        log.debug("Removing scheduled items:")
        self.print_scheduled_items(func_print=log.debug)
        self.items = []
        self.stop_realtime_jobs()
        self.realtime_items = {}
