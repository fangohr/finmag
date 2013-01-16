import logging
import argparse
import dolfin as df
from finmag.sim.sim_helpers import sim_with
from finmag.util import configuration, ansistrm
from finmag.util.helpers import start_logging_to_file

_FINMAG_LOG_LEVELS = {
        "EXTREMEDEBUG" : 5,
        "DEBUG" : logging.DEBUG,
        "INFO" : logging.INFO,
        "WARNING" : logging.WARNING,
        "ERROR" : logging.ERROR,
        "CRITICAL" : logging.CRITICAL
}

_DOLFIN_LOG_LEVELS = {
        "DEBUG" : df.DEBUG,
        "INFO" : df.INFO,
        "WARN" : df.WARNING,
        "WARNING" : df.WARNING,
        "ERROR" : df.ERROR,
        "CRITICAL" : df.CRITICAL,
        "PROGRESS" : df.PROGRESS,
}

# If no finmag configuration file exists, create a default one in ~/.finmagrc.
configuration.create_default_finmagrc_file()

# Read the settings from the configuration file first.
logfiles = configuration.get_config_option("logging", "logfiles", "").split()
finmag_level = _FINMAG_LOG_LEVELS[configuration.get_config_option("logging", "console_logging_level", "DEBUG")]
dolfin_level = _DOLFIN_LOG_LEVELS[configuration.get_config_option("logging", "dolfin_logging_level", "WARNING")]
color_scheme = configuration.get_config_option("logging", "color_scheme", "light_bg")

# Parse command line options.
parser = argparse.ArgumentParser(description='Parse the logging level and colour scheme.')
parser.add_argument("--verbosity", default=None,
        choices=("extremedebug", "debug", "info", "warning", "error", "critical"),
        help="Set the finmag logging level.")
parser.add_argument("--colour", default=None,
        choices=("dark_bg", "light_bg", "none"),
        help="Set the logging colour scheme.")
args, _ = parser.parse_known_args() # Parse only known args so py.test can parse the remaining ones.

# The parsed command line options can override the configuration file settings.
if args.verbosity:
    finmag_level = _FINMAG_LOG_LEVELS[args.verbosity.upper()]
if args.colour:
    color_scheme = args.colour

# Apply the settings.
df.set_log_level(dolfin_level) # use root logger
logger = logging.getLogger(name='finmag') # to control level separately from dolfin
logger.setLevel(logging.DEBUG)

ch = ansistrm.ColorizingStreamHandler()
ch.setLevel(finmag_level)

formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)

try:
    ch.level_map = ansistrm.level_maps[color_scheme]
except KeyError:
    raise ValueError("Unkown color scheme: '{}' (allowed values: {})".format(color_scheme, ansistrm.level_maps.keys()))

logger.addHandler(ch)

# Now add file handlers for all logfiles listed in the finmag configuration file.
for f in logfiles:
    start_logging_to_file(f, formatter=formatter, mode="a", level=finmag_level)
