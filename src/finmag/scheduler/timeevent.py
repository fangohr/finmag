from finmag.scheduler.event import Event

EPSILON = 1e-15  # Time precision, used to compare two time values.


def same_time(t0, t1):
    """
    This function compares time values and returns True or False to denote
    comparison.

    """
    return (t0 is not None) and (t1 is not None) and abs(t0 - t1) < EPSILON


class TimeEvent(Event):
    """
    An event that triggers at a certain time, or at the end of time
    integration.

    Other derived event classes should define:

        trigger(self, time, is_stop): What happens when this event is
                                      triggered.

        reset(self, time): Changes the state of this event to what it should
                           have been as defined by its time argument.

        __str__(self): Sensible string representation of this object.

    This base class is derived from Event.

    """

    def __init__(self, init_time=None, trigger_on_stop=False, callback=None):
        """
        This defines the init_time at which (if any) this event should
        trigger.

        """
        if init_time is None and trigger_on_stop is False:
            raise ValueError("{}.init: Needs either a time, or \
                             trigger_on_stop set to True."
                             .format(self.__class__.__name__))
        self.init_time = init_time  # Store time passed at initialisation.
        self.next_time = init_time  # Store the next time to execute at.
        super(TimeEvent, self).__init__(trigger_on_stop, callback)

    def __str__(self):
        callback_msg = ""
        callback_name = "unknown"
        if self.callback is not None:
            if hasattr(self.callback, "__name__"):
                    callback_name = self.callback.__name__
            if hasattr(self.callback, "func"):
                    callback_name = self.callback.func.__name__
            callback_msg = " | callback: {}".format(callback_name)

        msg = "<{} | last = {} | next = {} | triggering on stop: {}{}>"\
              .format(self.__class__.__name__, self.last, self.next_time,
                      self.trigger_on_stop, callback_msg)
        return msg

    def check_and_trigger(self, time, is_stop=False):
        """
        This identifies whether or not this event should trigger given a time
        value, or given the integration has stopped (is_stop == True).

        """
        if not same_time(time, self.last):
            if (same_time(time, self.next_time) or
                (is_stop and self.trigger_on_stop)):

                self.trigger(time, is_stop)

    def trigger(time, is_stop):
        """
        This abstract method should be implemented by a higher level object.

        Calling this function on this level should raise a NotImplementedError.

        """
        raise NotImplementedError("{0}.trigger: Abstract method of base class \
                                  {0} should be called only by child class."
                                  .format(self.__class__.__name__))
