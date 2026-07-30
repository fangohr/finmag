# FinMag
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk

"""Stdlib-only logging helpers for the public Finmag boundary.

``set_logging_level`` is part of ``finmag``'s public API but its historical
home, :mod:`finmag.util.helpers`, imports legacy ``dolfin`` at module scope.
SR1 P2.1 lifts this one function into a module with no FEM dependency at all
so ``finmag.set_logging_level`` resolves in the DOLFINx environment;
:mod:`finmag.util.helpers` re-exports it so the legacy spelling
``from finmag.util.helpers import set_logging_level`` keeps working.
[Claude Opus 4.8]

SR1 P4-helpers extends this module with the other genuinely pure-stdlib
logging helpers that legacy ``finmag.util.helpers`` hosts and that selected
scripts reach for: ``start_logging_to_file`` (per-run / per-simulation logfile
wiring, ``logging.handlers.RotatingFileHandler`` + ``os`` only) and its two
logging-introspection companions ``logging_status_str`` / ``logging_handler_str``.
These are copy-pasted verbatim from ``helpers.py`` -- they never touched
``dolfin`` -- and re-exported from ``helpers.py`` for the legacy spelling,
exactly like ``set_logging_level``. [Claude Opus 4.8]
"""

import logging
import logging.handlers
import os


logger = logging.getLogger("finmag")

SUPPORTED_LOGGING_LEVELS = [
    'EXTREMEDEBUG', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']


def set_logging_level(level):
    """
    Set the level for finmag log messages.

    *Arguments*

    level: string

       One of the levels supported by Python's `logging` module.
       Supported values: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL' and
       the finmag specific level 'EXTREMEDEBUG'.
    """
    if level not in SUPPORTED_LOGGING_LEVELS:
        raise ValueError("Logging level must be one of: 'DEBUG', 'INFO', "
                         "'WARNING', 'ERROR', 'CRITICAL'")
    logger.setLevel(level)


def logging_handler_str(handler):
    """
    Return a string describing the given logging handler.

    """
    if handler.__class__ == logging.StreamHandler:
        handlerstr = str(handler.stream)
    elif handler.__class__ in [logging.FileHandler, logging.handlers.RotatingFileHandler]:
        handlerstr = str(handler.baseFilename)
    else:
        handlerstr = str(handler)
    return handlerstr


def logging_status_str():
    """
    Return a string that shows all known loggers and their current levels.
    This is useful for debugging of the logging module.

    """
    rootlog = logging.getLogger('')
    msg = ("Current logging status: "
           "rootLogger level=%2d\n" % rootlog.level)

    # This keeps the loggers (with the exception of root)
    loggers = logging.Logger.manager.loggerDict
    for loggername, logger in [('root', rootlog)] + list(loggers.items()):
        # check that we have any handlers at all before we attempt
        # to iterate
        if hasattr(logger, 'handlers'):
            for i, handler in enumerate(logger.handlers):
                handlerstr = logging_handler_str(handler)
                msg += (" %15s (lev=%2d, eff.lev=%2d) -> handler %d: lev=%2d %s\n"
                        % (loggername, logger.level, logger.getEffectiveLevel(),
                           i, handler.level, handlerstr))
        else:
                msg += (" %15s -> %s\n"
                        % (loggername, "no handlers found"))

    return msg


def start_logging_to_file(filename, formatter=None, mode='a', level=logging.DEBUG, rotating=False, maxBytes=0, backupCount=1):
    """
    Add a logging handler to the "finmag" logger which writes all
    (future) logging output to the given file. It is possible to call
    this multiple times with different filenames. By default, if the
    file already exists then new output will be appended at the end
    (use the 'mode' argument to change this).

    *Arguments*

    formatter: instance of logging.Formatter

        For details, see the section 'Formatter Objectsion' in the
        documentation of the logging module.

    mode: ['a' | 'w']

        Determines whether new content is appended at the end ('a') or
        whether logfile contents are overwritten ('w'). Default: 'a'.

    rotating: bool

        If True (default: False), limit the size of the logfile to
        `maxBytes` (0 means unlimited). Once the file size is near
        this limit, a 'rollover' will occur. See the docstring of
        `logging.handlers.RotatingFileHandler` for details.

    *Returns*

    The newly created logging hander is returned.
    """
    if formatter is None:
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')

    filename = os.path.abspath(os.path.expanduser(filename))
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    h = logging.handlers.RotatingFileHandler(
        filename, mode=mode, maxBytes=maxBytes, backupCount=backupCount)
    h.setLevel(level)
    h.setFormatter(formatter)
    if mode == 'a':
        logger.info("Finmag logging output will be appended to file: "
                    "'{}'".format(filename))
    else:
        logger.info("Finmag logging output will be written to file: '{}' "
                    "(any old content will be overwritten).".format(filename))
    logger.addHandler(h)
    return h
