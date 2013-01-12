try:
    import pynotify
except ImportError:
    print "You need the Python bindings to libnotify for this example to work."
    print "On Ubuntu, install the package 'python-notify'."
    raise

from sim_with_scheduling import example_simulation

# This script will show a notification on the screen when the simulation completes.

# Initialise the notification system with our application name (this is
# required by the libnotify library) and create a notification.
pynotify.init("Finmag")
my_notification = pynotify.Notification("Simulation completed")

# The function we will add to the scheduler.
# It will display the current simulation time when the simulation completes.
def notify_when_done(sim, note):
    msg = "Integrated up to t = {} ns.".format(sim.t * 1e9)
    note.set_property("body", msg)
    note.show()

sim = example_simulation()

# Register the function with the scheduler.
sim.schedule(notify_when_done, args=[my_notification], at_end=True)

# Integrate for one nanosecond. When the integration is done, the
# notification will be shown.
sim.run_until(1e-9)
