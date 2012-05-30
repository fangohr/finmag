import logging
from finmag.util import configuration, ansistrm
import dolfin as df

_DOLFIN_LOG_LEVELS = {
    "DEBUG": df.DEBUG,
    "INFO": df.INFO,
    "WARN": df.WARNING,
    "WARNING": df.WARNING,
    "ERROR": df.ERROR,
    "CRITICAL": df.CRITICAL,
    "PROGRESS": df.PROGRESS,
}

# create logger
logger = logging.getLogger(name='finmag')

# Note: dolfin uses the default logger ('root'). Se we should 
# use a separate one to be able to
# control levels of details separately for finmag and dolfin.
# Here we setup this logger with name 'finmag'

#change this to get more detailed output
logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

def parse_logging_level(s, values=logging._levelNames):
    if s is None:
        return s
    try:
        return int(s)
    except ValueError:
        return values[s]

# create console handler; the logging level is read from the config file
ch = ansistrm.ColorizingStreamHandler()

# Read the logging settings from the configuration file
console_level = parse_logging_level(configuration.get_config_option("logging", "console_logging_level", logging.DEBUG))
ch.setLevel(console_level)
dolfin_level = parse_logging_level(configuration.get_config_option("logging", "dolfin_logging_level"), _DOLFIN_LOG_LEVELS)
if dolfin_level is not None:
    df.set_log_level(dolfin_level)

# create formatter #(some options to play with)
#formatter = logging.Formatter('%(asctime)s-%(levelname)s: %(message)s')
#formatter = logging.Formatter('L%(asctime)s-%(levelname)s: %(message)s')
#formatter = logging.Formatter('FL:%(relativeCreated)10.1f-%(levelname)s: %(message)s')
formatter = logging.Formatter('%(levelname)s: %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

#A first message
logger.debug("Executing sim/__init__.py")

#and examples to use in code later
#logger.debug("debug message")
#logger.info("info message")
#logger.warn("warning message")


