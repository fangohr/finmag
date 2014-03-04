
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
import os
import sys
import time
import subprocess
import textwrap
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


def create_index_html(conf):
    """
    Create an index.html file which contains a title and links to all
    the individual notebook html files.
    """
    # XXX TODO: This is just a quick hack for proof-of-concept. The
    #           resulting html could be made much nicer using
    #           nbconvert's internal python functions directly, but it
    #           would probably need a bit of tweaking for that.
    targetdirname = targetdirectoryname(conf)
    partner_str = ", ".join(conf['names'])

    logging.debug("[DDD] lst: {} (type: {})".format(conf['tutorials'], type(conf['tutorials'])))
    contents_str = \
        "<br>\n".join(map(lambda name: "   <li><a href='{f}'>{f}</a></li>".format(f=name+'.html'),
                          list(conf['tutorials'])))

    with open(os.path.join(targetdirname, 'index.html'), 'w') as f:
        f.write(textwrap.dedent("""\
            <h2>Finmag Manual for {partner_str}</h2>
            <br>
            Tutorial compiled on {date_str}.
            <br><br>
            Contents:<br>
            <ul>
               {contents_str}
            </ul>
            <br><br>
            The raw ipython notebooks are <a href='ipynb/'>available here</a>
            """.format(partner_str=partner_str,
                       date_str=time.asctime(),
                       contents_str=contents_str)))


def assemble_rst(conf):

    rstrootfilenames = conf['tutorials']
    targetdirname = targetdirectoryname(conf)

    for fileroot in rstrootfilenames:
        # copy all rst files (and where we have images also the corresponding _files directories)
        # to the target location
        shutil.copy("../ipython_notebooks_dest/%s.rst" % fileroot, os.path.join(targetdirname,fileroot+".rst"))
        if os.path.exists("../ipython_notebooks_dest/%s_files" % fileroot):
            targetdirnamefiles = os.path.join(targetdirname,fileroot+"_files")
            if os.path.exiscompile_ipynb2htmlts(targetdirnamefiles):
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
        f = open(os.path.jocompile_ipynb2htmlin(targetdirname,fileroot + '.rst'),'r')
        for line in f:
            fout.write(line)
        fout.write('\n\n')  # need white space before next section to have valid rst

    fout.close()


def compile_rst2html(conf):
    targetdirname = targetdirectoryname(conf)
    cmd = "cd %s; rst2html.py --stylesheet-path=../../../css/voidspace.css index.rst index.html" % targetdirname
    logging.debug("Running cmd '%s'" % cmd)
    output = subprocess.check_output(cmd, shell=True)
    logging.debug("Output was %s" % output)


def compile_ipynb2html(conf):
    #try:
    #    NBCONVERT = os.environ['NBCONVERT']
    #except KeyError:
    #    print("Please set the environment variable NBCONVERT so that it "
    #          "points to the location of the script nbconvert.py.")
    #    sys.exit(1)
    # XXX TODO: It would be much nicer to use the python functions in
    #           nbconvert directly rather than calling it via
    #           'subprocess'. Once the --output-dest flag (or a
    #           similar one) has been adopted into the main repo we
    #           should revisit this.

    targetdir = targetdirectoryname(conf)
    ipynbdir = os.path.join(targetdir, 'ipynb')
    if not os.path.exists(ipynbdir):
        os.mkdir(ipynbdir)
    for fileroot in conf['tutorials']:
        shutil.copy("../ipython_notebooks_src/{}.ipynb".format(fileroot), ipynbdir)
        NBCONVERT='NBCONVERT'
        cmd = "cd {}/..; ipython nbconvert --to html ipynb/{}.ipynb".format(
            ipynbdir, fileroot, fileroot)
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
    parser.add_argument('--use-ipynb2html', action='store_true', default=True, \
                         help='Use nbconvert directly to convert .ipynb files to .html.')

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

        # Do the work
        if args['use_ipynb2html']:
            logging.info("Creating index.html for %s" % partner)
            create_index_html(conf)
            logging.info("Converting .ipynb files to .html for %s" % partner)
            compile_ipynb2html(conf)
        else:
            logging.info("Assembling rst file for %s" % partner)
            assemble_rst(conf)
            logging.info("Compiling rst file for %s" % partner)
            compile_rst2html(conf)

        logging.info("Creating tar ball for  %s" % partner)
        assemble_tarballs(conf)

        if not args['keepbuild']:
            remove_builddir(conf)
