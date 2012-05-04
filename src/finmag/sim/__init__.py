import logging
from finmag.util import configuration

# create logger
logger = logging.getLogger(name='finmag')

# Note: dolfin uses the default logger ('root'). Se we should 
# use a separate one to be able to
# control levels of details separately for finmag and dolfin.
# Here we setup this logger with name 'finmag'

#change this to get more detailed output
logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

def parse_logging_level(s, default_value):
    if s is None:
        return default_value
    try:
        return int(s)
    except ValueError:
        return logging._levelNames[s]

# create console handler; the logging level is read from the config file
ch = logging.StreamHandler()
if configuration.get_configuration().has_section("logging"):
    console_level = parse_logging_level(configuration.get_configuration().get("logging", "console_logging_level"), logging.DEBUG)
else:
    console_level = logging.DEBUG

ch.setLevel(console_level)

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


