directories_to_ignore = ['build','__pycache__']

import os
import shutil
import argparse


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
            if filename != "__init__.py":
                # create empty init.py file
                #f = open(os.path.join(targetdir, "__init__.py"), "w")
                #f.close()
                print "skipping py-file %s as a so exists" % filename
                return  # don't copy that Python file because we have a .so

        if os.path.exists(path[:-3] + ".pyc"):
            if (
                ('test' not in filename) and 
                (filename != "__init__.py") and 
                (filename != "run_nmag_Eexch.py") # for a test that
                                                  # passes this
                                                  # filename to nsim
                ):
                print 'skipping py-file %s as suitable pyc exists' % filename
                return # don't copy any .pyc test-files

    elif filename.endswith('pyc'):
        if (('test' in filename) 
            or filename.startswith('__init__')
            or (os.path.exists(path[:-4] + ".so"))):

            print "Skipping pyc file (%s) as it is a test or init, or a .so exists" % filename
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
    print scandir('finmag')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy FinMag files to alternative location')
    parser.add_argument('destination-dir', type=str, help='The directory to copy FinMag files to')
    args = parser.parse_args()

    targetdir = vars(args)['destination-dir']

    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    distcp()
