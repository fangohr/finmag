import os
import sys
import logging
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
    return get_module_version('IPython')


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
    Determine and return the version of the given Debian package.

    This only works on Debian-derived systems (such as Debian
    or Ubuntu) as it internally calls 'dpkg -s' to determine
    the version number.

    If the package is installed, returns a string with the version
    number, otherwise returns None. Raises NotImplementedError if
    the version cannot be determined due to an unsupported system.
    """
    import subprocess
    import re

    supported_distros = ['Ubuntu', 'Debian', 'Linux Mint']
    linux_issue = get_linux_issue()
    version = None

    if any([d in linux_issue for d in supported_distros]):
        try:
            with open(os.devnull, 'w') as devnull:
                output = subprocess.check_output(['dpkg', '-s', pkg_name], stderr=devnull)
            lines = output.split('\n')
            version_str = filter(lambda s: s.startswith('Version'), lines)[0]
            version = re.sub('Version: ', '', version_str)
        except subprocess.CalledProcessError:
            pass
    else:
        raise NotImplementedError(
            "This does not seem to be a supported (i.e. Debian-derived) "
            "Linux distribution. Cannot determine version of package "
            "'{}'".format(pkg_name))

    return version


def get_version_sundials():
    return get_debian_package_version('libsundials-serial')


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


def get_version_netgen():
    import subprocess
    try:
        with open(os.devnull) as devnull:
            output = subprocess.check_output(['netgen', '-v'], stderr=devnull)
        version = output.split('\n')[0]
    except subprocess.CalledProcessError:
        version = None
    return version


def running_binary_distribution():
    """Return True if this is the cython-based binary 
    distribution or False if it is source distribtion
    """

    thefile = __file__

    if thefile.split('.')[1] in ['py', 'pyc']:
        #logger.debug("Running source code version")
        return False
    elif thefile.split('.')[1] == 'so':
        #logger.debug("Binary finmag distribution")
        return True
    else:
        logger.error("thefile=%s" % thefile)
    raise RuntimeError("This is impossible.")


if __name__ == "__main__":
    linux_issue = get_linux_issue()

    print("__file__ = %s" % __file__)
    print("Linux issue: %s" % linux_issue)
    print("Binary distribution: %s" % running_binary_distribution())
    print("Sundials version: %s" % get_version_sundials())
