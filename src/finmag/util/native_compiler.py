# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

"""
Automatic compilation of C/C++ extension modules.

Invoking make_modules() will run the Makefile in the native code directory.
If make returns an error code, an exception will be raised.

Only the first call to make_modules() will invoke make; subsequent calls are ignored.

This module should not be used directly. Use

    from finmag.native import [symbol]

when a native function or class is required.
"""
import subprocess
import logging
import sys
import os
import re

__all__ = ["make_modules"]

NATIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../native")
MODULES_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../finmag/native")
MAKEFILE = os.path.join(NATIVE_DIR, "Makefile")

logger = logging.getLogger("finmag")

def replace_c_errors_with_python_errors(s):
    repl = lambda m: r'File "%s", line %s (%s): ' % (os.path.abspath(os.path.join(NATIVE_DIR, m.group(1))), m.group(2), m.group(3))
    return re.sub(r"([^\s:]+):(\d+):(\d+): ", repl, s)

def run_make(cmd, **kwargs):
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, **kwargs)
    except subprocess.CalledProcessError, ex:
        output = replace_c_errors_with_python_errors(ex.output)
        with open(MODULES_OUTPUT_DIR + "/compiler_errors.log", "w") as f:
            f.write(output)
        print "If you can't see the error message below, either set your term to deal with utf8, or check the file src/finmag/native/compiler_errors.log"
        sys.stderr.write(output)
        raise Exception("make_modules: Make failed")

modules_compiled = False
def make_modules():
    global modules_compiled
    if not modules_compiled:
        if not os.environ.has_key('DISABLE_PYTHON_MAKE') and os.path.exists(MAKEFILE):
            # FIXME: The next line always prints, even if modules are built.
            # It may be possible to fix this by running 'make -q' first and
            # checking its exit status, but this seems to require some
            # restructuring of the build logic in 'native'.
            logger.debug("Building modules in 'native'...")
            run_make(["make"], cwd=NATIVE_DIR)
        modules_compiled = True

def pipe_output(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, bufsize=1)
    while True:
        line = process.stdout.readline()
        if not line:
            break
        print replace_c_errors_with_python_errors(line),
    process.communicate()
    return process.poll()
