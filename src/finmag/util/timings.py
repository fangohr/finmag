import os
import sys
import time
import functools
from contextlib import contextmanager

"""
Measure the time functions, methods, or pieces of code need to run.

This module provides the Timings class. An instance of Timings manages the
measurements and prints out a summary of the findings when printed:

    timer = Timings()
    # ...
    print timer

The module provides a default instance of the Timings class, which can be used
by importing the variable *timings* from this module.

    from finmag.util.timings import timings
    # ...
    print timings

There are four different ways to do a measurement.

1. Conduct a measurement by hand:

    timer = Timings()
    timer.start('my_measurement')
    sleep(1)
    timer.stop('my_measurement')

2. Time a piece of code using the context manager *timed*:

    timer = Timings()
    with timed('my_measurement', timer=timer):
        sleep(1)

    # or with the default timer *timings* (which is defined in this module)
    with timed('my_measurement'):
        sleep(1)

3. Time a function with the decorator *ftimed*:

    timer = Timings()

    @ftimed(timer)
    def do_things():
        sleep(1)

    # or with the default timer *timings* (which is defined in this module)

    @ftimed() # needs the parentheses
    def do_things()
        sleep(1)

4. Time a method with the decorator *mtimed*:

    timer = Timings()

    class Example(object):
        @mtimed(timer)
        def do_things(self):
            sleep(1)

    # or with the default timer *timings* (which is defined in this module)

    class Example(object):
        @mtimed() # needs the parentheses
        def do_things(self):
            sleep(1)

"""


class Timings(object):
    """
    Manage a series of time measurements.

    """
    def __init__(self):
        self.reset()
        self._last_started_measurement = None

    def reset(self):
        """
        Reset the internal data.

        The self._measurements dictioniary has the measurement names as keys
        and lists [n, t, st] as values, where n is the number of calls, t the
        cumulative time it took and st the status ('finished', STARTTIME).

        """
        self._measurements = {}
        self._creation_time = time.time()

    def start(self, name):
        """
        Start a measurement with the name *name*.

        """
        if name in self._measurements.keys():
            assert self._measurements[name][2] == 'finished', \
                "Still running measurement for '{}', can't start another one.".format(name)
            self._measurements[name][2] = time.time()
        else:
            self._measurements[name] = [0, 0., time.time()]
        self._last_started_measurement = name

    def stop(self, name):
        """
        Stop the measurement with the name *name*.

        """
        assert name in self._measurements.keys(), \
            "No known measurement with name '{}'. Known: {}.".format(
                    name, self._measurements.keys())

        assert self._measurements[name][2] != 'finished', \
                "Measurement for '{}' not running, can't stop it.".format(name)

        timetaken = time.time() - self._measurements[name][2]
        self._measurements[name][0] += 1
        self._measurements[name][1] += timetaken
        self._measurements[name][2] = 'finished'
        self._last_started_measurement = None

    def stoplast(self):
        """
        Stop the last started measurement.

        """
        assert self._last_started_measurement != None, "No measurement running, can't stop anything."
        self.stop(self._last_started_measurement)

    def startnext(self, name):
        """
        Stop the last started measurement to start a new one with name *name*.

        """
        if self._last_started_measurement:
            self.stop(self._last_started_measurement)
        self.start(name)

    def getncalls(self, name):
        return self._measurements[name][0]

    def gettime(self, name):
        return self._measurements[name][1]

    def report_str(self, nb_items=10):
        """
        Returns a listing of the *nb_items* measurements that ran the longest.
        
        """
        msg = "Timings summary, longest items first:\n"

        sorted_keys = sorted(
                self._measurements.keys(),
                key = lambda x: self._measurements[x][1],
                reverse=True)

        for i, name in enumerate(sorted_keys):
            nb_calls, total_time, _ = self._measurements[name]
            if nb_calls > 0:
                msg += "%35s:%6d calls took %10.4fs " \
                    "(%8.6fs per call)\n" % (name[0:35],
                                             self.getncalls(name),
                                             self.gettime(name),
                                             self.gettime(name)\
                                                 / float(self.getncalls(name))
                                             )
            else:
                msg = "Timings %s: none completed\n" % name

            if i >= nb_items - 1:
                break
        recorded_sum = self.recorded_sum()
        walltime = time.time() - self._creation_time
        msg += "Wall time: %.4gs (sum of time recorded: %gs=%5.1f%%)\n" % \
            (walltime, recorded_sum, recorded_sum / walltime * 100.0)

        return msg

    def __str__(self):
        return self.report_str()

    def recorded_sum(self):
        return sum([self._measurements[name][1] for name in self._measurements.keys()])

timings = Timings()

@contextmanager
def timed(name, timer=timings):
    """
    Use this context to time a piece of code. Needs a *name* argument for the measurement.

    """
    timer.start(name)
    yield
    timer.stop(name)

def mtimed(method_or_timer=timings):
    """
    Use to decorate a method to report timings. Accepts an optional Timings instance.

    """
    def decorator(method):
        name = method.__name__

        @functools.wraps(method)
        def decorated_method(that, *args, **kwargs):
            cls = that.__class__.__name__
            timer.start(name)
            method(that, *args, **kwargs)
            timer.stop(name)

        return decorated_method

    if callable(method_or_timer):
        # the user called mtimed without arguments, python thus calls it with
        # the method to decorate (which is callable). Use the default Timings
        # object and return the decorated method. 
        timer = timings
        return decorator(method_or_timer)
    # the user called mtimed with a timing object. Bind it to the timer name
    # and return the decorator itself. That's how python handles decorators which
    # take arguments.
    timer = method_or_timer
    return decorator

def ftimed(fn_or_timer=timings):
    """
    Use to decorate a function to report timings. Accepts an optional Timings instance.

    """
    def decorator(fn):
        name = fn.__name__
        filename = sys.modules[fn.__module__].__file__
        module = os.path.splitext(os.path.basename(filename))[0]

        @functools.wraps(fn)
        def decorated_function(*args, **kwargs):
            timer.start(name)
            fn(*args, **kwargs)
            timer.stop(name)

        return decorated_function
    
    if callable(fn_or_timer):
        # the user called ftimed without arguments, python thus calls it with
        # the method to decorate (which is callable). Use the default Timings
        # object and return the decorated function.
        timer = timings
        return decorator(fn_or_timer)
    # the user called ftimed with a timing object. Bind it to the timer name
    # and return the decorator itself. That's how python handles decorators which
    # take arguments.
    timer = fn_or_timer
    return decorator
