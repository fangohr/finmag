import os
import sys
import time
import operator
import functools
from collections import defaultdict
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


class SingleTiming(object):
    """
    Saves number of calls and total running time of a method.

    """
    def __init__(self, group, name):
        self.group = group
        self.name = name
        self.calls = 0
        self.tot_time = 0.0
        self.running = False
        self._start = 0.0

    def start(self):
        assert not self.running, \
            "Can't start running measurement '{}' of group '{}' again.".format(self.name, self.group)
        self.running = True
        self._start = time.time()

    def stop(self):
        assert self.running, \
            "Measurement '{}' of group '{}' not running, can't stop it.".format(self.name, self.group)
        self.running = False
        self.calls += 1
        self.tot_time += time.time() - self._start


class Timings(object):
    """
    Manage a series of measurements.
    
    They are stored by the measurements' group and name, which is module/class
    and function/method name.

    """
    def __init__(self):
        self.reset()
        self._last = None

    def reset(self):
        """
        Reset the internal data.

        """
        self._timings = {}
        self._created = time.time()

    def start(self, group, name):
        """
        Start a measurement with the group *group* and name *name*.

        """
        if self.key(group, name) in self._timings:
            self._timings[self.key(group, name)].start()
        else:
            timing = SingleTiming(group, name)
            timing.start()
            self._timings[self.key(group, name)] = timing
        self._last = self._timings[self.key(group, name)]

    def stop(self, group, name):
        """
        Stop the measurement with the group *group* and name *name*.

        """
        assert self.key(group, name) in self._timings, \
                "No known measurement '{}' of '{}' to stop.".format(name, group)
        self._timings[self.key(group, name)].stop()
        self._last = None

    def stop_last(self):
        """
        Stop the last started measurement.

        """
        assert self._last, "No measurement running, can't stop anything."
        self._last.stop()

    def start_next(self, group, name):
        """
        Stop the last started measurement to start a new one with *group* and *name*.

        """
        if self._last:
            self.stop_last()
        self.start(group, name)

    def calls(self, group, name):
        """
        Returns the number of calls to the method of *group* with *name*.

        """
        return self._timings[self.key(group, name)].calls

    def time(self, group, name):
        """
        Returns the total running time of the method of *group* with *name*.

        """
        return self._timings[self.key(group, name)].tot_time

    def report(self, max_items=10):
        """
        Returns a listing of the *max_items* measurements that ran the longest.
        
        """
        msg = "Timings: Showing the up to {} slowest items.\n\n".format(max_items)
        separator = "+--------------------+------------------------------+--------+------------+--------------+\n"
        msg += separator
        msg += "| {:18} | {:28} | {:>6} | {:>10} | {:>12} |\n".format("class/module", "name", "calls", "total (s)", "per call (s)")
        msg += separator
        msg_row = "| {:18} | {:28} | {:>6} | {:>10.3g} | {:>12.3g} |\n"
        shown = 0
        for t in sorted(
                self._timings.values(),
                key=operator.attrgetter('tot_time'),
                reverse=True):
            msg += msg_row.format(t.group, t.name, t.calls, t.tot_time, t.tot_time/t.calls)
            shown += 1
            if shown >= max_items:
                break
        msg += separator + "\n"

        msg += "Timings grouped by class or module.\n\n"
        separator = "+--------------------+----------+------+\n"
        msg += separator
        msg += "| {:18} | {:>8} | {:>4} |\n".format('class/module', 'time (s)', '%')
        msg += separator
        for group, tot_t, share in self.grouped_timings():
            msg += "| {:18} | {:>8.3g} | {:>4.2g} |\n".format(group, tot_t, share)
        msg += separator + "\n"

        msg += "Total wall time {:.4} s.\n".format(time.time() - self._created)
        return msg

    def grouped_timings(self):
        """
        Returns a list of tuples (group_name, total_time, share), sorted by decreasing total_time.
        It also describes either how much time went unaccounted for, or if there are multiple
        measurements at the same time at some point.

        """
        grouped_timings = defaultdict(float)
        for timing in self._timings.values():
            grouped_timings[timing.group] += timing.tot_time

        recorded_time = self.total_time()
        wall_time = time.time() - self._created

        grouped_timings = [(group, tot_t, 100*tot_t/wall_time) for group, tot_t in grouped_timings.iteritems()]

        diff = abs(recorded_time - wall_time)
        rel_diff = 100 * (1 - recorded_time/wall_time)
        rel_diff_desc = "redundant" if recorded_time > wall_time else "untimed"
        grouped_timings.append((rel_diff_desc, diff, rel_diff))

        grouped_timings = sorted(
                grouped_timings,
                key=lambda gt : gt[1],
                reverse=True)

        return grouped_timings

    def __str__(self):
        return self.report()

    def total_time(self):
        return sum([tim.tot_time for tim in self._timings.itervalues()])

    def key(self, group, name):
        return group + "::" + name

timings = Timings()

@contextmanager
def timed(group, name, timer=timings):
    """
    Use this context to time a piece of code. Needs a *name* argument for the measurement.

    """
    timer.start(group, name)
    yield
    timer.stop(group, name)

def mtimed(method_or_timer=timings):
    """
    Use to decorate a method to report timings. Accepts an optional Timings instance.

    """
    def decorator(method):
        name = method.__name__

        @functools.wraps(method)
        def decorated_method(that, *args, **kwargs):
            cls = that.__class__.__name__
            # temporary way of replicating existing behaviour until categories
            # are implemented
            timer.start(cls, name)
            ret = method(that, *args, **kwargs)
            timer.stop(cls, name)
            return ret

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
            timer.start(module, name)
            ret = fn(*args, **kwargs)
            timer.stop(module, name)
            return ret

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
