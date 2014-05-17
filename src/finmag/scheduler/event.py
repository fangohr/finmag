# Event states are as follows:
#     EV_ACTIVE: Event intends to trigger eventually if integration continues.
#     EV_DONE: Event does not intent to trigger again. Should be removed by the
#         scheduler.
#     EV_REQUESTS_STOP_INTEGRATION: Event wishes the scheduler to stop the
#         integration. This does not necessarily remove this event.

EV_ACTIVE, EV_DONE, EV_REQUESTS_STOP_INTEGRATION = range(3)


class Event(object):
    """
    This base class defines generic event objects from which all other
    event objects (should) inherit.

    Upon trigger, the callback function will be called with no argument passed
    to it. Other event objects will inherit from this, and should define:

        __init__(self): Should check whether a value is set such that this
            event may trigger at all. An event that doesn't trigger is pretty
            useless, but this class is not passed parameters that fully define
            this behaviour so it should be performed at a higher level.

        check_trigger(self, args): Checks the current state to see if this
            event should have triggered or not.

        trigger(self, args): What happens when this event is triggered.

        reset(self, args): Changes the state of this event to what it should
           have been, as defined by the arguments.

        __str__(self): Sensible string representation of this object.

    """

    def __init__(self, trigger_on_stop=False, callback=None):
        """
        This function defines whether or not this event will trigger at the end
        of the integration, and can also be passed a function to call upon
        the event triggering.

        """
        self.state = EV_ACTIVE  # The state of this event.
        self.trigger_on_stop = trigger_on_stop
        self.last = None  # The previous time or step value with which this
                          # event last triggered.

        if callback is not None:
            self.attach(callback)
        else:
            self.callback = None

    def attach(self, callback):
        """
        This function stores the function 'callback', which takes no arguments,
        and which will be called when this event is triggered.

        """
        if not hasattr(callback, "__call__"):
            raise ValueError("{}.attach: Argument should be callable."
                             .format(self.__class__.__name__))
        self.callback = callback
