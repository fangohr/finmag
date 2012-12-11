import logging
logger = logging.getLogger('finmag')

def get_linux_issue():
    try:
        f = open("/etc/issue")
    except IOError:
        log.error("Can't read /etc/issue -- this is odd?")
        raise RuntimeError("Cannot establish linux version")
    issue = f.readline()  # only return first line
    issue = issue.replace('\\l','')
    issue = issue.replace('\\n','')
    #logger.debug("Linux OS = '%s'" % issue)
    return issue.strip() # get rid of white space left and right

def get_version_ipython():
    try:
        import IPython
        return IPython.__version__
    except ImportError:
        return None

def get_version_dolfin():
    try:
        import dolfin
        return dolfin.__version__
    except ImportError:
        return None

def get_version_numpy():
    try:
        import numpy
        return numpy.__version__
    except ImportError:
        return None

def get_version_matplotlib():
    try:
        import matplotlib
        return matplotlib.__version__
    except ImportError:
        return None

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

    print("__file__ = %s" % __file__)
    print("Linux issue:" + get_linux_issue())
    print("Binary distribution: %s" % running_binary_distribution())


