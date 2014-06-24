import logging
from finmag.scheduler.timeevent import TimeEvent, same_time, EPSILON

# Import the possible states of events.
from finmag.scheduler.event import EV_ACTIVE, EV_DONE
from finmag.scheduler.event import EV_REQUESTS_STOP_INTEGRATION

log = logging.getLogger(name="finmag")


class SingleTimeEvent(TimeEvent):
    """
    A time-based event that triggers at a certain time, and/or at the end of
    time integration.

    """

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

    If a variant time progression is desired, a callable object can be passed
    instead of constant interval, which will be evaluated by the trigger. The
    callable object accepts only "self" as an argument, and so can operate
    on properties defined in this event.
    """

    def __init__(self, interval, init_time=None, trigger_on_stop=False,
                 callback=None):
        super(RepeatingTimeEvent, self).__init__(init_time or 0.,
                                                 trigger_on_stop, callback)

        # Negative intervals make us sad.
        if interval < 0:
            raise ValueError("{}.init: Proposed interval is negative; events "
                             "cannot occur in the past without the use of "
                             "reset.".format(self.__class__.__name__))
        self.interval = interval

    def trigger(self, time, is_stop=False):
        super(RepeatingTimeEvent, self).trigger(time=time, is_stop=is_stop)

        # Calculate next time to trigger.
        if not hasattr(self.interval, "__call__"):
            self.next_time = self.last + self.interval
        else:
            self.next_time = self.last + self.interval()

    def reset(self, time):
        """
        As with base classes, though it is important to note that if this event
        is reset to a time that is precisely when the event would trigger,
        then it should trigger again.
        """
        if not hasattr(self.interval, "__call__"):
            self.last = time - time % self.interval
            if time % self.interval < EPSILON:
                self.last -= self.interval
            self.next_time = self.last + self.interval

        else:
            msg = "Resetting in time is not well defined for repeated " +\
                  "events with non-constant interval."
            raise NotImplementedError(msg)
