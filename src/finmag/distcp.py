#files_to_ignore = ['llg.py']
directories_to_ignore = ['build']
targetdir = "/tmp/finmag"

import os
import shutil


def cp_file(sourcedir, filename, targetdir):
    #only relevant case is if we have a .so file for a given .py, then don't copy .py
    #if the .so file is __init__.py, we need to copy an empty __init__.py

    #if directory does not exist, create (might be empty in the end)
    if not os.path.exists(targetdir):
        print "Creating directory %s" % targetdir
        os.makedirs(targetdir)

    path = os.path.join(sourcedir, filename)
    if filename.endswith('.py'):
        if os.path.exists(path[:-3] + ".so"):
            if filename == "__init__.py":
                # create empty init.py file
                f = open(os.path.join(targetdir, "__init__.py"), "w")
                f.close()
            return  # don't copy that Python file because we have a .so
    elif filename.endswith('pyc'):
        print "Skipping pyc file (%s)" % filename
        return
    elif filename.endswith('c'):
        print "Skipping .c   file (%s)" % filename
        return
    print("Copying %s" % path)
    shutil.copyfile(path, os.path.join(targetdir, filename))


def scandir(dir, files=[]):
    for file_ in os.listdir(dir):
        path = os.path.join(dir, file_)
        #print "working %s / %s" % (dir, file_)
        if os.path.isfile(path):
            cp_file(dir, file_, os.path.join(targetdir, dir))
        elif os.path.isdir(path) and os.path.split(path) not in directories_to_ignore:
            scandir(path, files)
    return files


def distcp():
    print scandir('sim')


if __name__ == '__main__':
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    distcp()
