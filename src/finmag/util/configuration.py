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
