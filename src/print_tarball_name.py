import os
import sys
from finmag.util.helpers import binary_tarball_name

try:
    revision = sys.argv[1]
    suffix = sys.argv[2]
    destdir = sys.argv[3]
except IndexError:
    print "Usage: print_tarball_name REVISION SUFFIX DESTDIR"
    sys.exit(0)

name = binary_tarball_name(repo=os.curdir, revision=revision, suffix=suffix)

print(os.path.join(destdir, name))
