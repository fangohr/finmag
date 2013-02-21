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
by importing the variable *default_timer* from this module.

    from finmag.util.timings import default_timer
    # ...
    print default_timer

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

    # or with the default timer *default_timer* (which is defined in this module)
    with timed('my_measurement'):
        sleep(1)

3. Time a function with the decorator *ftimed* (for standalone functions):

    timer = Timings()

    @ftimed(timer)
    def do_things():
        sleep(1)

    # or with the default timer *default_timer* (which is defined in this module)

    @ftimed # works with or without parentheses
    def do_things()
        sleep(1)

4. Time a method with the decorator *mtimed* (for methods inside a class):

    timer = Timings()

    class Example(object):
        @mtimed(timer)
        def do_things(self):
            sleep(1)

    # or with the default timer *default_timer* (which is defined in this module)

    class Example(object):
        @mtimed() # parentheses are optional
        def do_things(self):
            sleep(1)

Timings of functions or methods will be grouped by module or class-name
respectively. To group the measurements you do by hand or using the
context-manager, you can pass a second string after the name of the measurement.

    with timed('my_measurement', 'FooBarFeature'):
        sleep(1)

    # or

    timer.start('my_measurement, 'FooBarFeature'):
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
            "Can't start running measurement '{}' of group '{}' again. You could call stop_last on the timings object.".format(self.name, self.group)
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
    default_group = "None"

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the internal data.

        """
        self._last = None
        self._timings = {}
        self._created = time.time()

    def start(self, name, group=default_group):
        """
        Start a measurement with the group *group* and name *name*.

        """
        if self.key(name, group) in self._timings:
            self._timings[self.key(name, group)].start()
        else:
            timing = SingleTiming(name, group)
            timing.start()
            self._timings[self.key(name, group)] = timing
        self._last = self._timings[self.key(name, group)]

    def stop(self, name, group=default_group):
        """
        Stop the measurement with the group *group* and name *name*.

        """
        assert self.key(name, group) in self._timings, \
            "No known measurement '{}' of '{}' to stop.".format(name, group)
        self._timings[self.key(name, group)].stop()
        self._last = None

    def stop_last(self):
        """
        Stop the last started measurement.

        """
        assert self._last, "No measurement running, can't stop anything."
        self._last.stop()

    def start_next(self, name, group=default_group):
        """
        Stop the last started measurement to start a new one with *group* and *name*.

        """
        if self._last:
            self.stop_last()
        self.start(name, group)

    def calls(self, name, group=default_group):
        """
        Returns the number of calls to the method of *group* with *name*.

        """
        return self._timings[self.key(name, group)].calls

    def time(self, name, group=default_group):
        """
        Returns the total running time of the method of *group* with *name*.

        """
        return self._timings[self.key(name, group)].tot_time

    def report(self, max_items=10):
        """
        Returns a listing of the *max_items* measurements that ran the longest.

        Warning: Due to the way this code works, letting time go by between
        running a simulation and calling this method will fudge the relative
        timings. This will likely be the case in interactive mode.

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
            msg += msg_row.format(t.group, t.name, t.calls, t.tot_time, t.tot_time / t.calls)
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

        seconds = time.time() - self._created
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        msg += "Total wall time %d:%02d:%02d." % (h, m, s)

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
        grouped_timings = [(group, tot_t, 100 * tot_t / wall_time) for group, tot_t in grouped_timings.iteritems()]

        diff = abs(recorded_time - wall_time)
        rel_diff = 100 * (1 - recorded_time / wall_time)
        rel_diff_desc = "redundant" if recorded_time > wall_time else "untimed"
        grouped_timings.append((rel_diff_desc, diff, rel_diff))

        grouped_timings = sorted(
            grouped_timings,
            key=lambda gt: gt[1],
            reverse=True)

        return grouped_timings

    def __str__(self):
        return self.report()

    def total_time(self):
        return sum([tim.tot_time for tim in self._timings.itervalues()])

    def key(self, name, group):
        return group + "::" + name

default_timer = Timings()


@contextmanager
def timed(name, group=Timings.default_group, timer=default_timer):
    """
    Use this context to time a piece of code. Needs a *name* argument for the measurement.

    """
    timer.start(name, group)
    yield
    timer.stop(name, group)


def mtimed(method_or_timer=default_timer):
    """
    Use to decorate a method to report timings. Accepts an optional Timings instance.

    """
    def decorator(method):
        name = method.__name__

        @functools.wraps(method)
        def decorated_method(self, *args, **kwargs):
            cls = self.__class__.__name__
            # temporary way of replicating existing behaviour until categories
            # are implemented
            timer.start(name, cls)
            ret = method(self, *args, **kwargs)
            timer.stop(name, cls)
            return ret

        return decorated_method

    if callable(method_or_timer):
        # the user called mtimed without arguments, python thus calls it with
        # the method to decorate (which is callable). Use the default Timings
        # object and return the decorated method.
        timer = default_timer
        return decorator(method_or_timer)
    # the user called mtimed with a timing object. Bind it to the timer name
    # and return the decorator itself. That's how python handles decorators which
    # take arguments.
    timer = method_or_timer
    return decorator


def ftimed(fn_or_timer=default_timer):
    """
    Use to decorate a function to report timings. Accepts an optional Timings instance.

    """
    def decorator(fn):
        name = fn.__name__
        filename = sys.modules[fn.__module__].__file__
        module = os.path.splitext(os.path.basename(filename))[0]

        @functools.wraps(fn)
        def decorated_function(*args, **kwargs):
            timer.start(name, module)
            ret = fn(*args, **kwargs)
            timer.stop(name, module)
            return ret

        return decorated_function

    if callable(fn_or_timer):
        # the user called ftimed without arguments, python thus calls it with
        # the method to decorate (which is callable). Use the default Timings
        # object and return the decorated function.
        timer = default_timer
        return decorator(fn_or_timer)
    # the user called ftimed with a timing object. Bind it to the timer name
    # and return the decorator itself. That's how python handles decorators which
    # take arguments.
    timer = fn_or_timer
    return decorator
