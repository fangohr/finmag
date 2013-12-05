# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk

from __future__ import division
import logging
logger = logging.getLogger("finmag")

from finmag.sim.sim import Simulation, sim_with, normal_mode_simulation
from finmag.util.helpers import set_logging_level
from finmag.util import configuration
import util.versions
from __version__ import __version__
import example
import signal

# Convenience access to physics object Q
from finmag.util.physics import Q


def timings_report(n=10):
    """
    Returns the timings report, which is an overview of where finmag's runtime is spent.

    By default, it will show the 10 functions where the most runtime has been spent.
    This number can be changed by passing an integer to this function.

    Usage:
        import finmag
        print finmag.timings_report()

    """
    from aeon import default_timer
    return default_timer.report(n)


logger.debug("%20s: %s" % ("Finmag", __version__))
display_module_versions = configuration.get_config_option("logging", "display_module_versions_at_startup", "True")
if display_module_versions == "True":
    logger.debug("%20s: %s" % ("Dolfin", util.versions.get_version_dolfin()))
    logger.debug("%20s: %s" % ("Matplotlib", util.versions.get_version_matplotlib()))
    logger.debug("%20s: %s" % ("Numpy", util.versions.get_version_numpy()))
    logger.debug("%20s: %s" % ("Scipy", util.versions.get_version_scipy()))
    logger.debug("%20s: %s" % ("IPython", util.versions.get_version_ipython()))
    logger.debug("%20s: %s" % ("Python", util.versions.get_version_python()))
    logger.debug("%20s: %s" % ("Paraview", util.versions.get_version_paraview()))
    try:
        sundials_version = util.versions.get_version_sundials()
    except NotImplementedError:
        sundials_version = '<cannot determine version>'
    logger.debug("%20s: %s" % ("Sundials", sundials_version))
    try:
        boost_version = util.versions.get_version_boostpython()
    except NotImplementedError:
        boost_version = '<cannot determine version>'
    logger.debug("%20s: %s" % ("Boost-Python", boost_version))
    logger.debug("%20s: %s" % ("Linux", util.versions.get_linux_issue()))


if util.versions.running_binary_distribution():
    # check that this is the same as the binary distribution has been compiled for
    # This matters for sundials: on 12.04 there is one version of sundials
    # on 12.10 there is a different one. They are not compatible, but we
    # have no way to tell which one we are using.
    #
    # We thus assume that we use the system's sundials, and thus we
    # should be able to check by comparing the linux distribution.
    import util.binary # Where is this module?
    logger.debug("%20s: %s" % ("Build Linux", util.binary.buildlinux))
    vb = util.binary.buildlinux
    vr = util.versions.get_linux_issue()
    if  vb == vr:
        logger.debug("Build Linux and host linux versions agree.")
    else:
        if util.versions.loose_compare_ubuntu_version(vb,vr):
            logger.warn("Build Linux and host linux versions only agree approximately.")
        else:
            logger.error("Build Linux = %s" % util.binary.buildlinux)
            logger.error("Host Linux = %s" % util.versions.get_linux_issue())
            raise RuntimeError("Build and Host linux must be identical, otherwise sundials may produce wrong results / crash")

# create extreme debugging logging level, which has numerical value 5
logging.EXTREMEDEBUG = 5
logging.addLevelName(logging.EXTREMEDEBUG, 'EXTREMEDEBUG')

# and register a function function for this for our logger
logger.extremedebug = lambda msg: logger.log(logging.EXTREMEDEBUG, msg)

# Register a function which starts the debugger when the program
# receives the 'SIGQUIT' signal (keyboard shortcut: "Ctrl-\").
def receive_quit_signal(signum, stack):
    print("Starting debugger. Type 'c' to resume execution and 'q' to quit.")
    try:
        # Try 'ipdb' first because it's nicer to use
        import ipdb; ipdb.set_trace()
    except ImportError:
        # Otherwise fall back to the regular 'pdb'.
        import pdb; pdb.set_trace()

logger.debug("Registering debug signal handler. Press Ctrl-Z at any time "
             "to stop execution and jump into the debugger.")
signal.signal(signal.SIGTSTP, receive_quit_signal)
