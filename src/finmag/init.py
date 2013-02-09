# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk

from __future__ import division
import logging
logger = logging.getLogger("finmag")

# XXX TODO:
#
# There is a strange incompatibility between paraview and Python's vtk
# module (which is imported by dolfin) which leads to a segfault if
# the line 'from paraview import servermanager' is executed *after*
# 'import vtk'. This is being investigated, but for now the workaround
# is to import paraview.servermanager before everything else. However,
# since this leads to problems with existing scripts if they contain
# the imports in the wrong order, we only try to import Paraview if
# this was explicitly requested by setting the following option in
# .finmagrc:
#
#     [misc]
#     import_paraview = True
#
from util.configuration import get_config_option
if get_config_option('misc', 'import_paraview') == 'True':
    try:
        from paraview import servermanager
    except ImportError:
        logging.warning("The .finmagrc config file contains the option "
                        "'import_paraview = True', but paraview is not "
                        "installed on this system.")

from finmag.sim.sim import Simulation, sim_with
from finmag.util.helpers import set_logging_level
from __version__ import __version__
import example

logger.debug("%20s: %s" % ("Finmag", __version__))
import util.versions
logger.debug("%20s: %s" % ("Dolfin", util.versions.get_version_dolfin()))
logger.debug("%20s: %s" % ("Matplotlib", util.versions.get_version_matplotlib()))
logger.debug("%20s: %s" % ("Numpy", util.versions.get_version_numpy()))
logger.debug("%20s: %s" % ("Scipy", util.versions.get_version_scipy()))
logger.debug("%20s: %s" % ("IPython", util.versions.get_version_ipython()))
logger.debug("%20s: %s" % ("Python", util.versions.get_version_python()))
logger.debug("%20s: %s" % ("Netgen", util.versions.get_version_netgen()))
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
    if util.binary.buildlinux == util.versions.get_linux_issue():
        logger.debug("Build Linux and host linux versions agree.")
    else:
        logger.error("Build Linux = %s" % util.binary.buildlinux)
        logger.error("Host Linux = %s" % util.versions.get_linux_issue())
        raise RuntimeError("Build and Host linux must be identical, otherwise sundials may produce wrong results / crash")

# create extreme debugging logging level, which has numerical value 5
logging.EXTREMEDEBUG = 5
logging.addLevelName(logging.EXTREMEDEBUG, 'EXTREMEDEBUG')

# and register a function function for this for our logger
logger.extremedebug = lambda msg: logger.log(logging.EXTREMEDEBUG, msg)
