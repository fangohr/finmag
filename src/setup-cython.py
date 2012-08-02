# build script for compiled version of finmag

# change this as needed
libincludedir = "."

import sys
import os

from distutils.core import setup
from distutils.extension import Extension

# we'd better have Cython installed, or it's a no-go
try:
    from Cython.Distutils import build_ext
except:
    print "You don't seem to have Cython installed. Please get a"
    print "copy from www.cython.org and install it"
    sys.exit(1)


# scan the directory for extension files, converting
# them to extension names in dotted notation
files_to_ignore = ['llg.py',
                   'bem_computation_tests.py',
                   'test_hello.py',
                   'native_compiler.py',
#                   'solver_base.py',        # abstract method
#                   'energy_base.py',        # abstract method
                   'oommf_calculator.py',   # oommf/test_mesh.py fails
                   'magpar.py',             # +3 failures
                   'consts.py',             # +37 failures when compiled.
                   'solid_angle_magpar.py', # +1 failure
                   'arraytools.py',         # +1 failure
                   'material.py',           # +1 failure
                   'solver_gcr.py',         # +2 failures
                   'helpers.py',            # +1 failure
                   '__init__.py',            # +2 failures (but only
                                            # finmag.sim.integrator.__init__),
                                            # not other __init__
                                            # files.
                   'test_mesh.py',          # py.test will not read a .so, no
                   'solver_fk_test.py'      # point compiling these two test-files.
                   ]

directories_to_ignore = ['tests']


def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".py") and \
            file not in files_to_ignore:
                files.append(path.replace(os.path.sep, ".")[:-3])
        elif os.path.isdir(path):
            thisdirectoryname = os.path.split(path)[1]
            if thisdirectoryname not in directories_to_ignore:
                scandir(path, files)
            else:
                print("skipping directory dir =%20s, path=%s" % (dir, path))
    return files


# generate an Extension object from its dotted name
def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep) + ".py"
    return Extension(
        extName,
        [extPath],
        include_dirs=[libincludedir, "."],   # adding the '.' to include_dirs
                                             # is CRUCIAL!!
        #extra_compile_args = ["-O3", "-Wall"],
        #libraries = ["dv",],
        )

# get the list of extensions
extNames = scandir("finmag")

# and build up the set of Extension objects
extensions = [makeExtension(name) for name in extNames]

print "extNames are\n", extNames
#print "extensions are\n", extensions

with open ('extension_names.txt','w') as ext_name_f:
    for ext_name in extNames:
        ext_name_f.write(ext_name + '\n')


# finally, we can pass all this to distutils
setup(
  name="finmag",
  packages=["finmag", "finmag.energies", "finmag.sim", 'finmag.util'],
  ext_modules=extensions,
  cmdclass={'build_ext': build_ext},
)
