

# need to move these configuration details into a separate file
tutorials = ['tutorial-example2','tutorial-saving-averages-demo','tutorial-use-of-logging']
conf = {'nameshort':'Hesjedahl',
        'names': ('Thorsten Hesjedahl','Shilei Zhang'),
        'header':".. contents::\n\n",
        'ipynbdir':"The raw ipython notebooks are `available here <ipynb>`__.\n\n" }

# library file starts

import tempfile
import shutil
import os.path
import time

def targetdirectoryname(conf):
    use_build = True
    if use_build == False:
        tmpdirname = tempfile.mkdtemp()
    else:
        tmpdirname = '_build'
        if not os.path.exists(tmpdirname):   # if we don't use
            os.mkdir(tmpdirname)
    targetdirname = os.path.join(tmpdirname, 'finmag-tutorial-%s' % conf['nameshort'])

    if os.path.exists(targetdirname):
            assert use_build == True # otherwise this should be impossible
    else:
        os.mkdir(targetdirname)

    return targetdirname

def assemble_rst(rstrootfilenames):

    targetdirname = targetdirectoryname(conf) 
        
    
    for fileroot in rstrootfilenames:
        # copy all rst files (and where we have images also the corresponding _files directories)
        # to the target location  
        shutil.copy("../ipython_notebooks_dest/%s.rst" % fileroot, os.path.join(targetdirname,fileroot+".rst"))
        if os.path.exists("../%s_files" % fileroot):
            targetdirnamefiles = os.path.join(targetdirname,fileroot+"_files")
            shutil.copytree("../ipython_notebooks_dest/%s_files" % fileroot, targetdirnamefiles)
            
            
        # also copy the raw ipynb files into the right directory
        ipynbdir = os.path.join(targetdirname, 'ipynb')
        if not os.path.exists(ipynbdir):
            os.mkdir(ipynbdir)
        shutil.copy("../ipython_notebooks_src/%s.ipynb" % fileroot, ipynbdir)
        
    
    # Create index file
    fout = open(os.path.join(targetdirname,'index.rst'),'w')
    
    title = "Finmag Manual for %s" % (",".join(conf['names']))
    fout.write("%s\n" % title)
    fout.write("=" * len(title) + '\n\n')

    fout.write("Tutorial compiled at %s.\n\n" % time.asctime())
    
    fout.write(conf['header'])
    
    fout.write(conf['ipynbdir'])
  
   
    # Assemble the index file
    for fileroot in rstrootfilenames:
        f = open(os.path.join(targetdirname,fileroot + '.rst'),'r')
        for line in f:
            fout.write(line)
        fout.write('\n\n')  # need white space before next section to have valid rst
    
    fout.close()
    

# <codecell>

def compile_rst2html():
    import subprocess
    targetdirname = targetdirectoryname(conf) 
    #output = os.system(")
    cmd = "cd %s; rst2html.py index.rst index.html" % targetdirname
    output = subprocess.check_output(cmd, shell=True)
    print ("Output = %s" % output)

# <codecell>

# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

tutorials = ['tutorial-example2','tutorial-saving-averages-demo','tutorial-use-of-logging']
conf = {'nameshort':'Hesjedahl',
        'names': ('Thorsten Hesjedahl','Shilei Zhang'),
        'header':".. contents::\n\n",
        'ipynbdir':"The raw ipython notebooks are `available here <ipynb>`__.\n\n" }
import tempfile
import shutil
import os.path
import time

# <codecell>

def targetdirectoryname(conf):
    use_build = True
    if use_build == False:
        tmpdirname = tempfile.mkdtemp()
    else:
        tmpdirname = '_build'
        if not os.path.exists(tmpdirname):   # if we don't use
            os.mkdir(tmpdirname)
    targetdirname = os.path.join(tmpdirname, 'finmag-tutorial-%s' % conf['nameshort'])

    if os.path.exists(targetdirname):
            assert use_build == True # otherwise this should be impossible
    else:
        os.mkdir(targetdirname)

    return targetdirname

def assemble_rst(rstrootfilenames):

    targetdirname = targetdirectoryname(conf) 
        
    
    for fileroot in rstrootfilenames:
        # copy all rst files (and where we have images also the corresponding _files directories)
        # to the target location  
        shutil.copy("../ipython_notebooks_dest/%s.rst" % fileroot, os.path.join(targetdirname,fileroot+".rst"))
        if os.path.exists("../%s_files" % fileroot):
            targetdirnamefiles = os.path.join(targetdirname,fileroot+"_files")
            shutil.copytree("../ipython_notebooks_dest/%s_files" % fileroot, targetdirnamefiles)
            
            
        # also copy the raw ipynb files into the right directory
        ipynbdir = os.path.join(targetdirname, 'ipynb')
        if not os.path.exists(ipynbdir):
            os.mkdir(ipynbdir)
        shutil.copy("../ipython_notebooks_src/%s.ipynb" % fileroot, ipynbdir)
        
    
    # Create index file
    fout = open(os.path.join(targetdirname,'index.rst'),'w')
    
    title = "Finmag Manual for %s" % (",".join(conf['names']))
    fout.write("%s\n" % title)
    fout.write("=" * len(title) + '\n\n')

    fout.write("Tutorial compiled at %s.\n\n" % time.asctime())
    
    fout.write(conf['header'])
    
    fout.write(conf['ipynbdir'])
  
   
    # Assemble the index file
    for fileroot in rstrootfilenames:
        f = open(os.path.join(targetdirname,fileroot + '.rst'),'r')
        for line in f:
            fout.write(line)
        fout.write('\n\n')  # need white space before next section to have valid rst
    
    fout.close()
    

def compile_rst2html():
    import subprocess
    targetdirname = targetdirectoryname(conf) 
    #output = os.system(")
    cmd = "cd %s; rst2html.py index.rst index.html" % targetdirname
    output = subprocess.check_output(cmd, shell=True)
    print ("Output = %s" % output)


if __name__ == '__main__':
    assemble_rst(tutorials)
    compile_rst2html()

