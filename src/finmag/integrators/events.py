"""
Defines different types of events for use with the scheduler.

They are expected to share the following attributes

    next: the next time the event should be triggered or None

    trigger_on_stop: whether the event should be triggered again at the
    end, when time integration stops

    requests_stop_integration: whether time integration should be stopped,

as well as the method

    trigger(self, time, is_stop): Where `time` should be equal to the `next`
    requested time, but can used for additional safety checks, and `is_stop`
    is False when time integration will continue after this step.

"""
import logging
from finmag.util.consts import ONE_DEGREE_PER_NS
from finmag.util.helpers import compute_dmdt

log = logging.getLogger(name="finmag")
EPSILON = 1e-15 # femtosecond precision for scheduling.

def same(t0, t1):
    return (t0 != None) and (t1 != None) and abs(t0 - t1) < EPSILON


class SingleEvent(object):
    """
    An event which will trigger once and optionally at the end of time integration.

    """
    def __init__(self, time=None, trigger_on_stop=False):
        """
        Define the time for which to trigger with the float `time` and/or
        specify if the event triggers at the end of time integration by
        setting `trigger_on_stop` to True or False.

        """
        if time == None and trigger_on_stop == False:
            raise ValueError("{}.init: Needs either a time, or trigger_on_stop set to True.".format(self.__class__.__name__))
        self._time = time
        self.last = None
        self.next = time
        self.trigger_on_stop = trigger_on_stop
        self._callback = None
        self.requests_stop_integration = False

    def attach(self, callback):
        """
        Save the function `callback` which should be callable without arguments.

        """
        if not hasattr(callback, "__call__"):
            raise ValueError("{}.attach: Argument should be callable.".format(self.__class__.__name__))
        self._callback = callback
        return self # so that object can be initialised and a function attached in one line
    call = attach # nicer name if assignment and init is indeed done in one line

    def trigger(self, time, is_stop=False):
        """
        Call the callback if the `time` is right.

        Will trigger if `time` is equal to the saved time, or if `is_stop` is
        True and trigger_on_stop was set to True as well. Won't trigger more
        than once per `time`, no matter what.

        If the callback returns False, the event will notify the scheduler
        to stop time integration.

        """
        if not same(time, self.last) and (
                (same(time, self.next)) or (is_stop and self.trigger_on_stop)):
            self.last = time
            self.next = self._compute_next()
            if self._callback != None:
                ret = self._callback()
                if ret == False:
                    self.requests_stop_integration = True

    def _compute_next(self):
        """
        Return the next target time. Will get called after the event is triggered.

        """
        return None # since this is a single event, there is no next step

    def reset(self, time):
        """
        Modify the internal state to what we would expect at `time`.

        """
        self.requests_stop_integration = False
        if time < self._time:
            self.last = None
            self.next = self._time
        else: # if we had to, assume we triggered for `time` already
            self.last = self._time
            self.next = None

    def __str__(self):
        """
        Return the informal string representation of this object.

        """
        callback_msg = ""
        callback_name = "unknown"
        if self._callback != None:
            if hasattr(self._callback, "__name__"):
                    callback_name = self._callback.__name__
            if hasattr(self._callback, "func"):
                    callback_name = self._callback.func.__name__
            callback_msg = " | callback: {}".format(callback_name)
        msg = "<{} | last = {} | next = {} | triggering on stop: {}{}>".format(
            self.__class__.__name__, self.last, self.next, self.trigger_on_stop,
            callback_msg)
        return msg


class RepeatingEvent(SingleEvent):
    """
    An event which will trigger in regular intervals and optionally at the end of time integration.

    """
    def __init__(self, interval, delay=None, trigger_on_stop=False):
        """
        Define the interval between two times for which to trigger with the
        float `interval`. The first execution can be set with the float `delay`.
        Specify if the event triggers at the end of time integration by
        setting `trigger_on_stop` to True or False.

        """
        super(RepeatingEvent, self).__init__(delay or 0.0, trigger_on_stop)
        self._interval = interval

    def _compute_next(self):
        """
        Return the next target time. Will get called after the event is triggered.

        """
        return self.last + self._interval

    def reset(self, time):
        """
        Modify the internal state to what we would expect at `time`.

        """
        self.last = None
        self.next = self._time
        while self.next <= time: # if we had to, assume we triggered for `time`
            self.last = self.next
            self.next = self._compute_next()


class StopIntegrationEvent(object):
    """
    Store the information that the simulation should be stopped at a defined time.

    """
    def __init__(self, time):
        """
        Define the `time` at which the simulation should be stopped.

        """
        self._time = time
        self.last = None
        self.next = self._time
        self.requests_stop_integration = False
        self.trigger_on_stop = False

    def trigger(self, time, is_stop=False):
        """
        If `time` is equal to the pre-defined time, request to stop time integration.

        """
        if same(time, self.next):
            self.last = time
            self.next = None
            self.requests_stop_integration = True

    def reset(self, time):
        """
        Modify the internal state to what we would expect at `time`.

        """
        if time < self._time:
            self.last = None
            self.next = self._time
            self.requests_stop_integration = False
        else:
            self.last = self._time
            self.next = None
            self.requests_stop_integration = True

    def __str__(self):
        """
        Return the informal string representation of this object.

        """
        return "<{} will stop time integration at t = {}>".format(
                self.__class__.__name__, self._next)


class RelaxationEvent(object):
    """
    Monitors the relaxation of the magnetisation over time.

    """
    def __init__(self, sim, stopping_dmdt=ONE_DEGREE_PER_NS, dmdt_increased_counter_limit=500, dt_limit=1e-10):
        """
        Initialise the relaxation with a Simulation object and stopping parameters.

        """
        self.dt = 1e-14
        self.dt_increment_multi = 1.5
        self.dt_limit = dt_limit

        self.stopping_dmdt = stopping_dmdt
        self.dmdt_increased_counter = 0
        self.dmdt_increased_counter_limit = dmdt_increased_counter_limit
        self.dmdts = [] # list of (t, max_dmdt) tuples

        self.sim = sim
        self.last_t = sim.t
        self.last_m = sim.m.copy()

        # to communicate with scheduler
        self.next = self.last_t + self.dt
        self.requests_stop_integration = False
        self.trigger_on_stop = False

    def trigger(self, t, is_stop=False):
        assert same(t, self.sim.t)
        if same(self.last_t, t):
            return
        if self.requests_stop_integration:
            log.error("Time integration continued even though relaxation has been reached.")

        t = self.sim.t
        m = self.sim.m.copy()

        if self.last_m != None:
            dmdt = compute_dmdt(self.last_t, self.last_m, t, m)
            self.dmdts.append((t, dmdt))

            if dmdt > self.stopping_dmdt:
                if self.dt < self.dt_limit / self.dt_increment_multi:
                    if len(self.dmdts) >= 2 and dmdt < self.dmdts[-2][1]:
                        self.dt *= self.dt_increment_multi
                else:
                    self.dt = self.dt_limit

                log.debug("At t={:.3g}, last_dmdt={:.3g} * stopping_dmdt, next dt={:.3g}.".format(
                    t, dmdt / self.stopping_dmdt, self.dt))
                self._check_interrupt_relaxation()
            else:
                log.debug("Stopping integration at t={:.3g}, with dmdt={:.3g}, smaller than threshold={:.3g}.".format(
                    t, dmdt, float(self.stopping_dmdt)))
                self.requests_stop_integration = True # hoping this gets noticed by the Scheduler

        self.last_t = t
        self.last_m = m
        self.next += self.dt

    def _check_interrupt_relaxation(self):
        """
        This is a backup plan in case relaxation can't be reached by normal means.
        Monitors if dmdt increases too ofen.

        """
        if len(self.dmdts) >= 2:
            if self.dmdts[-1][1] > self.dmdts[-2][1]:
                self.dmdt_increased_counter += 1
                log.debug("dmdt {} times larger than last time (counting {}/{}).".format(
                    self.dmdts[-1][1] / self.dmdts[-2][1],
                    self.dmdt_increased_counter,
                    self.dmdt_increased_counter_limit))

        if self.dmdt_increased_counter >= self.dmdt_increased_counter_limit:
            log.warning("Stopping time integration after dmdt increased {} times.".format(
                self.dmdt_increased_counter_limit))
            self.requests_stop_integration = True


    def reset(self, time):
        pass # TODO: Will this "just work"?

    def __str__(self):
        """
        Return the informal string representation of this object.

        """
        return "<{} | last t = {} | last dmdt = {} * stopping_dmdt | next t = {}>".format(
                self.__class__.__name__, self.last_t, self.dmdts[-1][1]/self.stopping_dmdt, self.next)
