import sys
import logging
from sim_with_scheduling import example_simulation

log = logging.getLogger(name="finmag")
log.setLevel(logging.ERROR) # To better show output of this program.

# This script will update a progress bar on the screen while the simulation runs.

sim = example_simulation()

# Sets up the progress bar.

toolbar_width = 42
print "\nRunning simulation. Please wait..."
sys.stdout.write("[{}]".format(" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width-1)) # return to start of line, after '['

# Moves the arrow towards the right.

def update_bar(s):
    sys.stdout.write("\b\b--->")
    sys.stdout.flush()

# Register the arrow moving function and run the simulation.

sim.schedule(update_bar, every=0.5e-10)
sim.run_until(1e-9)

sys.stdout.write("\b-] Done.\n")
