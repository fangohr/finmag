
#
## need to move these configuration details into a separate file
#tutorials = ['tutorial-example2','tutorial-saving-averages-demo','tutorial-use-of-logging']
#conf = {'nameshort':'Hesjedahl',
#        'names': ('Thorsten Hesjedahl','Shilei Zhang'),
#        'header':".. contents::\n\n",
#        'ipynbdir':"The raw ipython notebooks are `available here <ipynb>`__.\n\n" }
#
# library file starts

import logging
logging.basicConfig(level = logging.DEBUG)
import tempfile
import shutil
import os.path
import time
import subprocess
from argparse import ArgumentParser

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

def assemble_rst(conf):

    rstrootfilenames = conf['tutorials']
    targetdirname = targetdirectoryname(conf) 
    
    for fileroot in rstrootfilenames:
        # copy all rst files (and where we have images also the corresponding _files directories)
        # to the target location  
        shutil.copy("../ipython_notebooks_dest/%s.rst" % fileroot, os.path.join(targetdirname,fileroot+".rst"))
        if os.path.exists("../ipython_notebooks_dest/%s_files" % fileroot):
            targetdirnamefiles = os.path.join(targetdirname,fileroot+"_files")
            if os.path.exists(targetdirnamefiles):
                shutil.rmtree(targetdirnamefiles)
            shutil.copytree("../ipython_notebooks_dest/%s_files" % fileroot, targetdirnamefiles)
            
        # also copy the raw ipynb files into the right directory
        ipynbdir = os.path.join(targetdirname, 'ipynb')
        if not os.path.exists(ipynbdir):
            os.mkdir(ipynbdir)
        shutil.copy("../ipython_notebooks_src/%s.ipynb" % fileroot, ipynbdir)
        
    
    # Create index file
    fout = open(os.path.join(targetdirname,'index.rst'),'w')
    
    title = "Finmag Manual for %s" % (", ".join(conf['names']))
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

def compile_rst2html(conf):
    targetdirname = targetdirectoryname(conf) 
    cmd = "cd %s; rst2html.py --stylesheet-path=../../../css/voidspace.css index.rst index.html" % targetdirname
    logging.debug("Running cmd '%s'" % cmd)
    output = subprocess.check_output(cmd, shell=True)
    logging.debug("Output was %s" % output)

def assemble_tarballs(conf):
    targetdirname = targetdirectoryname(conf) 
    cmd = "cd %s; tar cfvz finmag-tutorial-%s.tgz finmag-tutorial-%s" % \
       (os.path.join(targetdirname, '..'), conf['nameshort'], conf['nameshort'])
    logging.debug("Running cmd '%s'" % cmd)
    output = subprocess.check_output(cmd, shell=True)
    logging.debug("Output was %s" % output)
    logging.info("Tarball %s.tgz is located in _build" % conf['nameshort'])

def remove_builddir(conf):
    targetdirname = targetdirectoryname(conf) 
    cmd = shutil.rmtree(targetdirname)
    logging.debug("Removing %s" % targetdirname)


if __name__ == '__main__':
    #ArgumentParser
    from argparse import ArgumentParser
    parser=ArgumentParser(description="""Create tailored tutorial bundles""")
    parser.add_argument('partner', type=str, nargs='+', \
                         help='partner[s] to process (names of subdirectories)')
    parser.add_argument('--keepbuild', action='store_true', default=False, \
                         help='Keep directory in which tar ball is built (for debugging).')

    args = vars(parser.parse_args())
    
    if isinstance(args['partner'],list):
        partners = args['partner']
        logging.info("About to process %s" % ", ".join(partners))
    else:
        partners = [args['partner']]

    for partner in partners:
        if partner[-3:] == '.py':
            partner = partner[:-3]
        conffile = __import__(partner)
        conf = conffile.conf
        #add defaults
        conf['header'] = ".. contents::\n\n"
        conf['ipynbdir']  = "The raw ipython notebooks are `available here <ipynb>`__.\n\n" 

        # do the work
        logging.info("Assembling rst file for %s" % partner)
        assemble_rst(conf)
        logging.info("Compiling rst file for %s" % partner)
        compile_rst2html(conf)
        logging.info("Creating tar ball for  %s" % partner)
        assemble_tarballs(conf)
        if args['keepbuild']:
            pass
        else:
            remove_builddir(conf)
