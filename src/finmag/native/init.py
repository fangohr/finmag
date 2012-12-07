
"""
Provides the symbols from the finmag native extension module.

Importing this module will first compile the finmag extension module, then
load all of its symbols into this module's namespace.
"""
from ..util.native_compiler import make_modules as _make_modules
_make_modules()
