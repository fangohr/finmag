import logging
from finmag.scheduler.timeevent import TimeEvent, same_time

# Import the possible states of events.
from finmag.scheduler.event import EV_ACTIVE, EV_DONE
from finmag.scheduler.event import EV_REQUESTS_STOP_INTEGRATION

log = logging.getLogger(name="finmag")


class SingleTimeEvent(TimeEvent):
    """
    A time-based event that triggers at a certain time, and/or at the end of
    time integration.

    """

    def __init__(self, init_time=None, trigger_on_stop=False, callback=None):
        # These arguments are not passed as kwargs because single-line syntax
        # is encouraged for tidyness in the scheduler.
        super(SingleTimeEvent, self).__init__(init_time, trigger_on_stop,
                                              callback)

    def trigger(self, time, is_stop=False):
        """
        This calls the callback function now, and does not check whether it is
        correct to do so (this is the job of check_trigger).

        This method updates time values, executes the callback function, and
        may alter the state of this event.

        """
        self.last = time
        self.next_time = None

        if self.callback is None:
            log.warning("Event triggered with no callback function.")
        else:
            returnValue = self.callback()
            if returnValue is True:
                self.state = EV_DONE
            if returnValue is False:
                self.state = EV_REQUESTS_STOP_INTEGRATION

    def reset(self, time):
        """
        This changes the state of this event to what it should have been at
        the specified time.

        This does not re/set the callback function.

        """

        # Reset to initial if specified time is prior to initial time.
        if time < self.init_time:
            self.last = None
            self.next_time = self.init_time
            self.state = EV_ACTIVE

        # Otherwise, we have already triggered.
        else:
            self.last = self.init_time
            self.next_time = None
            self.state = EV_DONE


class RepeatingTimeEvent(SingleTimeEvent):
    """
    A time-based event that triggers regularly, and/or at the end of time
    integration.

    """

    def __init__(self, interval, init_time=None, trigger_on_stop=False,
                 callback=None):
        super(RepeatingTimeEvent, self).__init__(init_time or 0.,
                                                 trigger_on_stop, callback)
        self.interval = interval

    def trigger(self, time, is_stop=False):
        super(RepeatingTimeEvent, self).trigger(time=time, is_stop=is_stop)
        self.next_time = self.last + self.interval

    def reset(self, time):
        """
        As with base classes, though it is important to note that if this event
        is reset to a time that is precisely when the event would trigger,
        then it should trigger again.
        """
        self.last = time - time % self.interval
        if time % self.interval == 0:
            self.last -= self.interval
        self.next_time = self.last + self.interval


class StopIntegrationTimeEvent(SingleTimeEvent):
    """
    A time-based event that stops time integration at a given time value.

    """

    def __init__(self, init_time):
        def callback():
            return False
        super(StopIntegrationTimeEvent, self).__init__(init_time=init_time,
                                                       callback=callback)


# MV: This class is from the good old days, where the simulation object has a
# bit of a god complex. Be sure to change this to fit the new paradigm. <!>

from finmag.util.consts import ONE_DEGREE_PER_NS
from finmag.util.helpers import compute_dmdt


class RelaxationEvent(object):
    """
    Monitors the relaxation of the magnetisation over time.

    """
    def __init__(self, sim, stopping_dmdt=ONE_DEGREE_PER_NS,
                 dmdt_increased_counter_limit=500, dt_limit=1e-10):
        """
        Initialise the relaxation with a Simulation object and stopping
        parameters.

        """
        self.dt = 1e-14
        self.dt_increment_multi = 1.5
        self.dt_limit = dt_limit

        self.stopping_dmdt = stopping_dmdt
        self.dmdt_increased_counter = 0
        self.dmdt_increased_counter_limit = dmdt_increased_counter_limit
        self.dmdts = []  # list of (t, max_dmdt) tuples
        self.energies = []

        self.sim = sim
        self.last_t = sim.t
        self.last_m = sim.m.copy()

        # to communicate with scheduler
        self.next_time = self.last_t + self.dt
        self.state = EV_ACTIVE
        self.trigger_on_stop = False

    def check_and_trigger(self, t, is_stop=False):
        assert same_time(t, self.sim.t)
        if same_time(self.last_t, t):
            return
        if self.state == EV_REQUESTS_STOP_INTEGRATION:
            log.error("Time integration continued even though relaxation has "
                      "been reached.")

        t = self.sim.t
        m = self.sim.m.copy()

        if self.last_m is not None:
            dmdt = compute_dmdt(self.last_t, self.last_m, t, m)
            self.dmdts.append((t, dmdt))
            energy = self.sim.total_energy()
            self.energies.append(energy)

            if dmdt > self.stopping_dmdt:
                if self.dt < self.dt_limit / self.dt_increment_multi:
                    if len(self.dmdts) >= 2 and dmdt < self.dmdts[-2][1]:
                        self.dt *= self.dt_increment_multi
                else:
                    self.dt = self.dt_limit

                log.debug("At t={:.3g}, last_dmdt={:.3g} * stopping_dmdt, "
                          "next dt={:.3g}."
                          .format(t, dmdt / self.stopping_dmdt, self.dt))
                self._check_interrupt_relaxation()
            else:
                log.debug("Stopping integration at t={:.3g}, with dmdt={:.3g},"
                          " smaller than threshold={:.3g}."
                          .format(t, dmdt, float(self.stopping_dmdt)))
                # hoping this gets noticed by the Scheduler
                self.state = EV_REQUESTS_STOP_INTEGRATION

        self.last_t = t
        self.last_m = m
        self.next_time += self.dt

    def _check_interrupt_relaxation(self):
        """
        This is a backup plan in case relaxation can't be reached by normal
        means.

        Monitors if dmdt increases too ofen.

        """
        if len(self.dmdts) >= 2:
            if (self.dmdts[-1][1] > self.dmdts[-2][1] and
                self.energies[-1] > self.energies[-2]):

                self.dmdt_increased_counter += 1
                log.debug("dmdt {} times larger than last time "
                          "(counting {}/{})."
                          .format(self.dmdts[-1][1] / self.dmdts[-2][1],
                                  self.dmdt_increased_counter,
                                  self.dmdt_increased_counter_limit))

        if self.dmdt_increased_counter >= self.dmdt_increased_counter_limit:
            log.warning("Stopping time integration after dmdt increased {} "
                        "times without a decrease in energy "
                        "(which indicates that something might be wrong)."
                        .format(self.dmdt_increased_counter_limit))
            self.state = EV_REQUESTS_STOP_INTEGRATION

    def reset(self, time):
        pass  # TODO: Will this "just work"?

    def __str__(self):
        """
        Return the informal string representation of this object.

        """
        return "<{} | last t = {} | last dmdt = {} * stopping_dmdt | next t = {}>".format(
            self.__class__.__name__, self.last_t, self.dmdts[-1][1] / self.stopping_dmdt, self.next_time)

# Renaming for easy testing.
SingleEvent = SingleTimeEvent
RepeatingEvent = RepeatingTimeEvent
StopIntegrationEvent = StopIntegrationTimeEvent
