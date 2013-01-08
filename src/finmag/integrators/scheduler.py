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
    def __init__(self, interval, start=0.0):
        """
        Initialise with the interval between correct times and optionally, a starting time.

        """
        self.next_step = start
        self.interval = interval
        self.callback = None

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

    def call(self, func, at=None, every=None):
        """
        Add a function ``func`` that should get called either at the time passed
        in ``at`` or every ``every`` seconds.

        The function will be called without arguments.

        """
        assert (at==None and every!=None) or (at!=None and every==None)

        if at:
            at_item = At(at).call(func)
            self._add(at_item)

        if every:
            every_item = Every(every).call(func)
            self._add(every_item)

    def _add(self, at_or_every):
        self.items.append(at_or_every)

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
