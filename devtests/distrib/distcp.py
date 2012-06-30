import os
import shutil
import logging

#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
log = logging

#files_to_ignore = ['llg.py']
directories_to_ignore = ['build']
targetdir = "/tmp/finmag-build"
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
                # create empty init.py file
                f = open(os.path.join(targetdir, "__init__.py"), "w")
                f.close()
            return  # don't copy that Python file because we have a .so
        elif os.path.exists(path[:-3] + ".pyc"):  # second best: we have a pyc file
            #shutil.copyfile(path, path[:-3] + ".pyc")  # should copy the file
            #return  # don't copy that Python file because we have a .pyc
            pass
        else:
            log.warning("Warning: copying %20s to %s" % targetdir)
    elif filename.endswith('pyc'):
        log.debug("Skipping pyc file (%s)" % path)
        log.info("Copying %s to %s" % (path, targetpath))
        shutil.copyfile(path, targetpath)

    elif filename.endswith('.c'):
        log.debug("Skipping .c   file (%s)" % path)
        return
    elif filename.endswith('.o'):
        log.debug("Skipping .o   file (%s)" % path)
        return
    targetpath = os.path.join(targetdir, filename)
    log.info("Copying %s to %s" % (path, targetpath))
    shutil.copyfile(path, targetpath)


def scandir(srcdir, targetdir, files=[]):
    for file_ in os.listdir(srcdir):
        path = os.path.join(srcdir, file_)
        log.debug("scandir: working %s / %s" % (srcdir, file_))
        if os.path.isfile(path):
            cp_file(srcdir, file_, targetdir)
        elif os.path.isdir(path) and os.path.split(path) not in directories_to_ignore:
            scandir(path, targetdir, files)
    return files


def distcp(srcdirectory, targetdir):
    return scandir(srcdirectory, targetdir)


if __name__ == '__main__':
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    print(distcp(sourcedir, targetdir))
