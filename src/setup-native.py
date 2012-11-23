#!/usr/bin/env python

# Run native_compiler.make_modules() as part of the distribution
# process.

# See
# http://stackoverflow.com/questions/279237/python-import-a-module-from-a-folder
# for info on this addition before the import line, which works for modules in
# arbitrary directories (here, "finmag/util")

import os, sys, inspect

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"finmag/util")))
if cmd_subfolder not in sys.path:
     sys.path.insert(0, cmd_subfolder)

import native_compiler

native_compiler.make_modules()
