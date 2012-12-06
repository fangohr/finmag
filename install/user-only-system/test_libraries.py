def test_dolfin():
    import dolfin
    print("Found dolfin %s" % dolfin.__version__)

def test_scipy():
    import scipy
    print("Found scipy %s" % scipy.__version__)

def test_numpy():
    import numpy
    print("Found numpy %s" % numpy.__version__)

def test_matplotlib():
    import matplotlib
    print("Found matplotlib %s" % matplotlib.__version__)

def test_ipython():
    import IPython
    print("Found Ipython %s" % IPython.__version__)


def report_module_presence_and_version(modulename):
    try:
        mod = __import__(modulename)
    except ImportError:
        return 'missing (ImportError)'
    return mod.__version__

if __name__ == "__main__": # run as python program
    import sys
    print("%15s -> %s" % ("Python",sys.version.split()[0]))
    for modulename in ['IPython', 'numpy',
                       'matplotlib', 'scipy',
                       'dolfin']:
        print("%15s -> %s" % (modulename,
                            report_module_presence_and_version(modulename)))

