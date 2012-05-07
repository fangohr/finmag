import ConfigParser as configparser
import os

__all__ = ["get_configuration"]

CONFIGURATION_FILES = [
    os.path.expanduser("~/.finmagrc"),
    os.path.expanduser("~/.finmag/finmagrc")
]

_parser = configparser.SafeConfigParser()
_parser.read(CONFIGURATION_FILES)

def get_configuration():
    return _parser

def get_config_option(section, name, default_value=None):
    try:
        return get_configuration().get(section, name)
    except configparser.NoOptionError:
        return default_value
    except configparser.NoSectionError:
        return default_value
