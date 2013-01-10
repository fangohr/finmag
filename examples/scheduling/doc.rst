Using the Scheduling System
===========================

The Simulation class exposes a flexible scheduling system so that you can
provide custom functions to get executed during the simulation. This system
is exposed through the :py:meth:`.Simulation.schedule` method which takes your
function as the first argument. The scheduler will call it without arguments.

Events Defined in Simulation Time
---------------------------------

By default the schedule operates on simulation time expressed in
seconds. Use either the `at` keyword argument to define a single point
in time at which your function is called, or the `every` keyword to
specify an interval between subsequent calls to your function::

    # Using a Simulation instance called sim
    # and a custom function called my_fun.

    # Schedule my_fun to be called at t = 0.5 nanoseconds.
    sim.schedule(my_fun, at=0.5e-9)

    # Schedule my_fun to be called every 50 picoseconds.
    sim.schedule(my_fun, every=50e-12)

When specifying the interval, you can optionally use the `after` keyword to
delay the first execution of your function::

    # Delay the start with the 'after' keyword.
    sim.schedule(my_fun, every=50e=12, after=100e-12)

Additionally, you can define events using the `at_end=True` option to have
your function be executed at the end of the simulation::

    # every 100 picoseconds including at the end of simulation
    sim.schedule(my_fun, every=100e-12, at_end=True)

    # at the end of simulation, but not before
    sim.schedule(my_fun, at_end=True)

As you can see, this works in combination with `at` and `every` or on its own.

Events Defined in Real Time
---------------------------

You can also schedule actions using real time instead of simulation
time by setting the `realtime` option to `True`::

    # every 10 minutes
    sim.schedule(my_fun, every=600, realtime=True)

    # every 10 minutes starting in one hour
    sim.schedule(my_fun, every=600, after=3600, realtime=True)

Using realtime, you can use the `after` keyword on its own as well::

    # in 10 minutes
    sim.schedule(my_fun, after=600, realtime=True)

While the interval in `every` should always be expressed in seconds, `at` 
expects a `date or datetime <http://docs.python.org/2/library/datetime.html>`_,
or a string parseable as date or datetime. The argument `after` works
with either::

    # at four o'clock today
    import datetime
    now = datetime.datetime.now()
    today_at_four = datetime.combine(now.date(), datetime.time(16))
    sim.schedule(my_fun, at=today_at_four, realtime=True)

    # every hour starting in one hour
    sim.schedule(my_fun, every=3600, after=3600, realtime=True)

    # every 24 hours starting at midnight
    sim.schedule(my_fun, every=24*3600, after="2013-01-30 00:00:00", realtime=True)

If you use a string to pass in a datetime (and not simply a date) as shown in
the last example, don't forget to include the seconds.
