import os
import shutil
import logging
import commands

#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
log = logging

#files_to_ignore = ['llg.py']
directories_to_ignore = ['build', '__pycache__']
targetdir = "/tmp/finmag-build/finmag"
sourcedir = os.path.realpath("../../src")


def cp_file(sourcedir, filename, targetdir):
    log.debug("cp_file: src=%s, filename=%s, targetdir=%s" % (sourcedir, filename, targetdir))
    #only relevant case is if we have a .so file for a given .py, then don't copy .py
    #if the .so file is __init__.py, we need to copy an empty __init__.py

    #if directory does not exist, create (might be empty in the end)
    if not os.path.exists(targetdir):
        log.info("Creating directory %s" % targetdir)
        os.makedirs(targetdir)

    path = os.path.join(sourcedir, filename)
    if filename.endswith('.py'):
        if os.path.exists(path[:-3] + ".so"):  # best scenario: we have a .so file
            if filename == "__init__.py":
                pass
                # create empty init.py file
                #f = open(os.path.join(targetdir, "__init__.py"), "w")
                #f.close()
            return  # don't copy that Python file because we have a .so
        elif os.path.exists(path[:-3] + ".pyc"):  # second best: we have a pyc file
            shutil.copyfile(path[:-3] + ".pyc", os.path.join(targetdir, filename + "c"))
            return  # don't copy that Python file because we have a .pyc
        else:
            log.warning("Warning: copying %20s to %s" % (path, targetdir))
    elif filename.endswith('pyc'):
        targetpath = os.path.join(targetdir, filename)
        log.info("Copying .pyc %s to %s" % (path, targetpath))
        shutil.copyfile(path, targetpath)
        return
    elif filename.endswith('.c'):
        log.debug("Skipping .c   file (%s)" % path)
        return
    elif filename.endswith('.o'):
        log.debug("Skipping .o   file (%s)" % path)
        return
    targetpath = os.path.join(targetdir, filename)
    log.debug("Copying %s to %s" % (path, targetpath))
    shutil.copyfile(path, targetpath)


def scandir(srcdir, targetdir, files=[]):
    for file_ in os.listdir(srcdir):
        path = os.path.join(srcdir, file_)
        log.debug("scandir: working %s / %s" % (srcdir, file_))
        if os.path.isfile(path):
            cp_file(srcdir, file_, targetdir)
        elif os.path.isdir(path) and os.path.split(path)[1] not in directories_to_ignore:
            scandir(path, os.path.join(targetdir, file_), files)
    return files


def distcp(srcdirectory, targetdir):
    return scandir(srcdirectory, targetdir)


if __name__ == '__main__':
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    cmd = "cd " + sourcedir + " && python -m compileall src/finmag"
    (exitstatus, outtext) = commands.getstatusoutput(cmd)
    if exitstatus:
        print("Error occured: output=%s" % outtext)
        raise RuntimeError()
    print(outtext)

    print(distcp(sourcedir, os.path.join(targetdir, '')))

    """The go to target directory

    cd /tmp/finmag-build/finmag
    export PYTHONPATH=`pwd`
    and (i) try to import finmag
    and if this works
    (ii) run regression tests

    cd finmag
    py.test -v

    It seems that py.test cannot find the tests if they are not called .pyc
    """