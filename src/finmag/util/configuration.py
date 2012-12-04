import ConfigParser as configparser
import os

__all__ = ["get_configuration"]

CONFIGURATION_FILES = [
    os.path.expanduser("~/.finmagrc"),
    os.path.expanduser("~/.finmag/finmagrc")
]


def get_configuration():
    _parser = configparser.SafeConfigParser()
    _parser.read(CONFIGURATION_FILES)
    return _parser


def get_config_option(section, name, default_value=None):
    try:
        return get_configuration().get(section, name)
    except configparser.NoOptionError:
        return default_value
    except configparser.NoSectionError:
        return default_value


def write_finmagrc_template_to_file(filename):
    """
    Write some default finmag configuration options to the given file.
    """
    with open(filename, 'w') as f:
        f.write(FINMAGRC_TEMPLATE)

def create_default_finmagrc_file():
    """
    Check whether a configuration file already exists at any of the
    supported locations. If this is not the case, the file
    '~/.finmagrc' is created with some default configuration options.

    Supported locations for configuration files are:

        ~/.finmagrc
        ~/.finmag/finmagrc
    """
    import logging
    logger = logging.getLogger("finmag")

    if not any([os.path.exists(f) for f in CONFIGURATION_FILES]):
        try:
            write_finmagrc_template_to_file(os.path.expanduser('~/.finmagrc'))
            logger.info(
                "Created default configuration in '~/.finmagrc' because no "
                "Finmag configuration file was found. Please review the "
                "settings and adapt them to your liking.")
        except IOError as e:
            logger.info(
                "Could not create default configuration file '~/.finmagrc' "
                "(reason: {}). Please create one manually.".format(e.strerror))


# Template for the '.finmagrc' file. This is used in the documentation
# and to create a default configuration file if none exists yet.
FINMAGRC_TEMPLATE = \
"""\
[logging]

# Color_scheme choices: [dark_bg, light_bg, none]
color_scheme = dark_bg

# Logfiles entries:
#
# - Files with an absolute path name (such as '~/.finmag/global.log')
#   define global logfiles to which all finmag programs will add
#   log statements.
#
# - Filenames without an absolute path (such as 'session.log') result in
#   a logfile of that name being created in the current working
#   directory when the finmag module is loaded with 'import finmag'.
#
#  For example:
#
#logfiles =
#   ~/.finmag/global.log
#   session.log

logfiles =
    ~/.finmag/global.log

# Useful logging level choices: [DEBUG, INFO, WARNING]
console_logging_level = DEBUG
"""
