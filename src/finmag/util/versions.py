import os
import sys
import logging
import finmag
logger = logging.getLogger('finmag')

def get_linux_issue():
    try:
        f = open("/etc/issue")
    except IOError:
        logger.error("Can't read /etc/issue -- this is odd?")
        raise RuntimeError("Cannot establish linux version")
    issue = f.readline()  # only return first line
    issue = issue.replace('\\l','')
    issue = issue.replace('\\n','')
    #logger.debug("Linux OS = '%s'" % issue)
    return issue.strip() # get rid of white space left and right


def get_version_python():
    version = sys.version.split(' ')[0]
    assert version.count('.') == 2, "Unknown version format: %s" % version
    return version


def get_module_version(name):
    try:
        m = __import__(name)
        return m.__version__
    except ImportError:
        return None


def get_version_ipython():
    try:
        return get_module_version('IPython')
    except ValueError:
        # This is needed due to a strange error seen in some test runs:
        #
        #    /usr/lib/python2.7/dist-packages/IPython/utils/io.py:32: in __init__
        #    >               raise ValueError("fallback required, but not specified")
        #    E               ValueError: fallback required, but not specified
        #
        # It seems that this can happen because standard output is caught by
        # py.test, but providing the -s switch didn't help either.
        return None


def get_version_dolfin():
    return get_module_version('dolfin')


def get_version_numpy():
    return get_module_version('numpy')


def get_version_matplotlib():
    return get_module_version('matplotlib')


def get_version_scipy():
    return get_module_version('scipy')


def get_version_boostpython():
    """
    Determine and return the boost-python version.

    We check the name of the symlink of libboost_python.

    If libboost_python.so is installed, returns a string with the version
    number, otherwise returns None. Raises NotImplementedError if
    the version cannot be determined. This may mean the file is not available,
    or not available in the standard place (/usr/lib).
    """

    # get version number as string
    maj, min_, rev = get_version_python().split('.')


    # libfile = /usr/lib/libboost_python-py27.so' or similar 
    libfile = '/usr/lib/libboost_python-py%s%s.so' % (maj, min_)

    try:
        filename = os.readlink(libfile)
    except OSError:
        raise NotImplementedError(
            "Cannot locate %s. Cannot determine boost-python version." % libfile)

    # expect filename to be something like 'libboost_python-py27.so.1.49.0'
    version = filename.split(".so.")[1]
    
    return version


def get_debian_package_version(pkg_name):
    """
    Determine and return the version of the given Debian package (as a string).

    This only works on Debian-derived systems (such as Debian or Ubuntu) as
    it internally calls 'dpkg -s' to determine the version number.

    If the package is installed, returns a string with the version number,
    otherwise returns None. Warns if the version cannot be determined due to
    an unsupported system.
    """
    import subprocess
    import re

    version = None

    try:
        with open(os.devnull, 'w') as devnull:
            output = subprocess.check_output(['dpkg', '-s', pkg_name], stderr=devnull)
    except subprocess.CalledProcessError as e:
        logger.warning("Could not determine version of {} using dpkg.".format(pkg_name))
        if e.returncode == 1:
            logger.warning("The package {} is probably not installed.".format(pkg_name))
        elif e.returncode == 127:
            logger.warning("This does not seem to be a debian-derived Linux distribution.")
        else:
            logger.warning("Can't determine cause of error.")
        return None

    lines = output.split('\n')
    version_str = filter(lambda s: s.startswith('Version'), lines)[0]
    version = re.sub('Version: ', '', version_str)
    return version


def get_version_sundials():
    return finmag.native.sundials.get_sundials_version()


def get_version_paraview():
    # XXX TODO: There should be a more cross-platform way of
    # determining the Paraview version, but the only method I could
    # find is in the thread [1], and it doesn't work any more for
    # recent versions of Paraview. It's quite annoying that something
    # as simple as "import paraview; paraview.__version__" doesn't
    # work...
    #
    # [1] http://blog.gmane.org/gmane.comp.science.paraview.user/month=20090801/page=34
    return get_debian_package_version('paraview')


def running_binary_distribution():
    """Return True if this is the cython-based binary 
    distribution or False if it is source distribtion
    """

    thefile = __file__

    if thefile.endswith('.py') or thefile.endswith('.pyc'):
        #logger.debug("Running source code version")
        return False
    elif thefile.endswith('.so'):
        #logger.debug("Binary finmag distribution")
        return True
    else:
        logger.error("thefile=%s" % thefile)
    raise RuntimeError("Checking running_binary_distribution failed!")

def loose_compare_ubuntu_version(v1,v2):
    
    if not v1.startswith('Ubuntu') or not v2.startswith('Ubuntu'):
        return False

    from distutils.version import LooseVersion
    t1 = LooseVersion(v1).version
    t2 = LooseVersion(v2).version
    
    if t1[3] == t2[3] and t1[4] == t2[4]:
        return True
    
    return False
    


if __name__ == "__main__":
    linux_issue = get_linux_issue()
    
    print("__file__ = %s" % __file__)
    print("Linux issue: %s" % linux_issue)
    print("Binary distribution: %s" % running_binary_distribution())
    print("Sundials version: %s" % get_version_sundials())
    
    print loose_compare_ubuntu_version('Ubuntu 12.04.1 LTS', "Ubuntu 12.04.2 LTS")
