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
        self.target = time
        self.last_step = None
        self.next_step = self.target
        self.callback = None
        self.at_end = at_end
        self.stop_simulation = False

    def __str__(self):
        return "<At t={}>".format(self.target, ", at_end=True" if self.at_end else "")

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

    def reset(self, current_time):
        self.stop_simulation = False
        if current_time >= self.target:
            self.last_step = self.target
            self.next_step = None
        else:
            self.last_step = None
            self.next_step = self.target


class Every(At):
    """
    Store a function that should be run and a time interval between function calls.
    Call that function if notified that the timing is right.

    """
    def __init__(self, interval, start=None, at_end=False):
        self.last_step = None
        self.first_step = start or 0.0
        self.next_step = self.first_step
        self.interval = interval
        self.callback = None
        self.at_end = at_end
        self.stop_simulation = False

    def __str__(self):
        return "<Every {} seconds>".format(
            self.interval,
            ", start={}".format(self.first_step) if self.first_step != 0.0 else "",
            ", at_end=True" if self.at_end else "")

    def update(self):
        self.next_step += self.interval

    def reset(self, current_time):
        self.next_step = self.first_step
        while self.next_step <= current_time:
            self.update()


class ExitAt(object):
    """
    Store the information that the simulation should be stopped at a defined time.

    """
    def __init__(self, time):
        self.target = time
        self.next_step = self.target
        self.at_end = False
        self.stop_simulation = False

    def __str__(self):
        return "<ExitAt t={}>".format(self.target)

    def fire(self, time):
        assert abs(time - self.next_step) < EPSILON
        self.next_step = None
        self.stop_simulation = True

    def reset(self, current_time):
        if current_time > self.next_step:
            self.stop_simulation = True
            self.next_step = None
        else:
            self.stop_simulation = False
            self.next_step = self.target


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
        assert not (at!=None and every!=None), "It's either `at` or `every`."
        assert not (at!=None and after!=None), "Delays don't mix with `at`."

        args = args or []
        kwargs = kwargs or {}
        callback = functools.partial(func, *args, **kwargs)

        if realtime:
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
            if item.stop_simulation == True:
                raise StopIteration
            if item.next_step != None and (next_step == None or next_step > item.next_step):
                next_step = item.next_step

        if next_step == None:
            raise StopIteration

        if next_step <= self.last:
            log.error("Scheduler computed the next time step should be t = {:.2g} s, but the last one was already t = {:.2g} s.".format(next_step, self.last))
            raise ValueError("Scheduler is corrupted. Requested a time step in the past: dt = {:.2g}.".format(next_step-self.last))
        return next_step

    def reached(self, time):
        """
        Notify the Scheduler that a certain point in time has been reached.

        It will perform the action(s) that were defined before.

        """
        for item in self.items:
            if (item.next_step != None) and abs(item.next_step - time) < EPSILON:
                item.fire(time)
        self.last = time

    def finalise(self, time):
        """
        Trigger all events that need to happen at the end of time integration.

        """
        for item in self.items:
            if item.at_end:
                item.fire(time)

    def reset(self, time):
        """
        Override schedule so that internal time is now `time` and modify scheduled items accordingly.

        """
        self.last = None
        for item in self.items:
            item.reset(time)

    def _print_item(self, item, func_print=log.info):
        func_print("'{}': {}".format(item.callback.func.__name__, item))

    def print_scheduled_items(self, func_print=log.info):
        for item in self.items:
            self._print_item(item, func_print)
        func_print("XXX TODO: Print realtime items, too")

    def clear(self):
        log.debug("Removing scheduled items:")
        self.print_scheduled_items(func_print=log.debug)
        self.items = []
        self.stop_realtime_jobs()
        self.realtime_items = []

def save_ndt(sim):
    """Given the simulation object, saves one line to the ndt finalise
    (through the TableWriter object)."""
    if sim.driver == 'cvode':
        log.debug("Saving data to ndt file at t={} (sim.name={}).".format(sim.t, sim.name))
    else:
        raise NotImplementedError("Only cvode driver known.")
    sim.tablewriter.save()
