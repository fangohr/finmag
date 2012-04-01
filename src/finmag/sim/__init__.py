import logging
# create logger
logger = logging.getLogger(name='finmag')

# Note: dolfin uses the default logger ('root'). Se we should 
# use a separate one to be able to
# control levels of details separately for finmag and dolfin.
# Here we setup this logger with name 'finmag'

#change this to get more detailed output
logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

# create console handler and set level to info
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

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


