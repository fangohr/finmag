import atexit
import logging
import functools
from numbers import Number
from datetime import datetime, timedelta
from finmag.scheduler.derivedevents import SingleEvent, RepeatingEvent
from finmag.scheduler.timeevent import same_time
from finmag.scheduler.event import EV_DONE, EV_REQUESTS_STOP_INTEGRATION
# This module will try to import the package apscheduler when a realtime event
# is added. Install with "pip install apscheduler".
# See http://pypi.python.org/pypi/APScheduler for documentation.

log = logging.getLogger(name="finmag")


class Scheduler(object):
    """
    Manages a list of actions that should be performed at specific times.

    Note that this class *intentionally* contains hardly any error
    checking. The correct behaviour of the Scheduler depends on the
    parent code doing "the right thing". In particular, it is crucial
    that the reached() method be called with the next time step at
    which an event is expected to happen, which can be obtained using
    the next() method.

    Thus a typical (correct) usage is as follows:

        s = Scheduler()
        s.add(...)       # schedule some item(s)
        t = s.next()     # get next time step at which something should happen

        # [do some stuff based on the time step just obtained]

        s.reached(t)

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

    def add(self, func, args=None, kwargs=None, at=None, at_end=False,
            every=None, after=None, realtime=False):
        """
        Register a function with the scheduler.

        Returns the scheduled item, which can be removed again by
        calling Scheduler._remove(item). Note that this may change in
        the future, so use with care.

        """
        if not hasattr(func, "__call__"):
            raise TypeError("The function must be callable but object '%s' is of type '%s'" %
                            (str(func), type(func)))
        assert at or every or at_end or (after and realtime), "Use either `at`, `every` or `at_end` if not in real time mode."
        assert not (at is not None and every is not None), "Cannot mix `at` with `every`. Please schedule separately."
        assert not (at is not None and after is not None), "Delays don't mix with `at`."

        args = args or []
        kwargs = kwargs or {}
        callback = functools.partial(func, *args, **kwargs)

        if realtime:
            if at_end:
                at_end_item = SingleEvent(None, True, callback)
                self._add(at_end_item)
            return at_end_item

        if at or (at_end and not every):
            at_item = SingleEvent(at, at_end, callback)
            self._add(at_item)
            return at_item

        if every:
            every_item = RepeatingEvent(every, after, at_end, callback)
            self._add(every_item)
            return every_item

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

        Automatically called upon iteration of scheduler instance.

        """
        next_step = None

        for item in self.items:
            if item.state == EV_REQUESTS_STOP_INTEGRATION:
                raise StopIteration
            if item.next_time is not None and (next_step is None or next_step > item.next_time):
                next_step = item.next_time

        if next_step is None:
            raise StopIteration

        if next_step < self.last:
            log.error("Scheduler computed the next time step should be t = {:.2g} s, but the last one was already t = {:.2g} s.".format(next_step, self.last))
            raise ValueError("Scheduler is corrupted. Requested a time step in the past: dt = {:.2g}.".format(next_step - self.last))
        return next_step

    def reached(self, time):
        """
        Notify the Scheduler that a certain point in time has been reached.

        It will perform the action(s) that were defined to happen at that time.

        """
        for item in self.items:
            if same_time(item.next_time, time):
                item.check_and_trigger(time)
                if item.state == EV_DONE:
                    self._remove(item)
        self.last = time

    def finalise(self, time):
        """
        Trigger all events that need to happen at the end of time integration.

        """
        for item in self.items:
            if item.trigger_on_stop:
                item.check_and_trigger(time, is_stop=True)

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
            print item  # this will call __str__ on the item, which should be defned for all events
        for item in self.realtime_items:
            self._print_realtime_item(item, func_print)

    def clear(self):
        log.debug("Removing scheduled items:")
        self.print_scheduled_items(func_print=log.debug)
        self.items = []
        self.stop_realtime_jobs()
        self.realtime_items = {}

    def run(self, integrator, callbacks_at_scheduler_events=[]):
        """
        Integrate until an exit condition in the schedule has been met.

        The optional argument `callbacks_at_scheduler_events` should be a
        list of functions which are called whenever the time integration
        reaches a "checkpoint" where some event is scheduled. Each such
        function should expect the timestep t at which the event occurs as
        its single argument. Note that these functions are called just
        *before* the scheduled events are triggered. This is used, for
        example, to keep time-dependent fields up to date with the
        simulation time.

        """
        self.start_realtime_jobs()

        for t in self:
            assert(t >= integrator.cur_t)  # sanity check

            # If new items were scheduled after a previous time
            # integration finished, we can have t == integrator.cur_t.
            # However, this confuses the integrators so we don't integrate
            # in this case.
            if t != integrator.cur_t:
                integrator.advance_time(t)

            for f in callbacks_at_scheduler_events:
                f(t)
            self.reached(t)

        self.finalise(t)
        self.stop_realtime_jobs()
