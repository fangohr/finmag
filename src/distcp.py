import os
import shutil
import argparse

directories_to_ignore = ['build', '__pycache__']


def cp_file(sourcedir, filename, targetdir):
    # The only relevant case is if we have a .so file for a given .py,
    # then don't copy .py if the .so file is __init__.py, we need to
    # copy an empty __init__.py

    # Create directory if it does not exist (it might be empty in the end)
    if not os.path.exists(targetdir):
        print "Creating directory %s" % targetdir
        os.makedirs(targetdir)

    path = os.path.join(sourcedir, filename)

    if filename.endswith('.py'):
        if os.path.exists(path[:-3] + ".so"):
            if filename != "__init__.py":
                # Create empty init.py file
                #f = open(os.path.join(targetdir, "__init__.py"), "w")
                #f.close()
                print "skipping py-file %s as a so exists" % filename
                return  # don't copy that Python file because we have a .so

        if os.path.exists(path[:-3] + ".pyc"):
            if (
                ('test' not in filename) and
                (filename != "__init__.py") and
                (filename != "run_nmag_Eexch.py")  # for a test that
                                                   # passes this
                                                   # filename to nsim
                ):
                print 'skipping py-file %s as suitable pyc exists' % filename
                return  # don't copy any .pyc test-files

    elif filename.endswith('pyc'):
        if (('test' in filename)
            or filename.startswith('__init__')
            or (os.path.exists(path[:-4] + ".so"))):

            print("Skipping pyc file ({}) as it is a test or init, "
                  "or a .so exists".format(filename))
            return

    elif filename.endswith('c'):
        print "Skipping .c   file (%s)" % filename
        return

    print("Copying %s" % path)
    shutil.copyfile(path, os.path.join(targetdir, filename))


def scandir(srcdir, files=[]):
    for file_ in os.listdir(srcdir):
        path = os.path.join(srcdir, file_)
        #print "working %s / %s" % (srcdir, file_)
        if os.path.isfile(path):
            cp_file(srcdir, file_, os.path.join(targetdir, srcdir))
        elif (os.path.isdir(path) and
              os.path.split(path) not in directories_to_ignore):
            scandir(path, files)
    return files


def distcp(targetdir):
    print scandir('finmag', targetdir)



def get_linux_issue():
    """Same code as in finmag.util.version - 
    it is difficult (at the moment) to re-use the existing code as we need 
    to use this function to create the finmag/util/binary.py file. If 
    we import finmag to access this function, it will try to check for the binary.py
    file which are just trying to create.

    This should be possible to detangle once we have a working setup..."""

    try:
        f = open("/etc/issue")
    except IOError:
        print "Can't read /etc/issue -- this is odd?"
        raise RuntimeError("Cannot establish linux version")
    issue = f.readline()  # only return first line
    issue = issue.replace('\\l','')
    issue = issue.replace('\\n','')
    #logger.debug("Linux OS = '%s'" % issue)
    return issue.strip() # get rid of white space left and right



def storeversions(targetfile):
    """Target file should be something like 'finmag/util/binary.py'
    The data in the file is used to store which version of software we 
    had when the binary distribution was created."""
    if os.path.exists(targetfile):
        print("This is odd: the file '%s' exists already, but is only" % targefile)
        print("meant to be created now (in function storeversions() in distcp.py)")
        raise RuntimeError("odd error when running %s" % __file__)

    f = open(targetfile,'w')
    f.write("buildlinux = '%s'\n" % get_linux_issue())
    f.close()


if __name__ == '__main__':
    descr = 'Copy FinMag files to alternative location'
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('destination-dir', type=str,
                        help='The directory to copy FinMag files to')
    args = parser.parse_args()

    targetdir = vars(args)['destination-dir']

    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    distcp(targetdir)

    storeversions(os.path.join(targetdir,"finmag/util/binary.py"))

