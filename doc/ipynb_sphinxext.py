'''
Sphinx/docutils extension to convert .ipynb files to .html files and
create links to the generated html in the source.

To use it, include something like the following in your .rst file:

   :ipynb:`path/to/notebook.ipynb`


XXX TODO: Currently this only works for the 'html' builder (not for
          singlehtml or latexpdf)
'''

import shutil
import os
import sys
import re
from IPython.nbformat import current as nbformat


def get_notebook_title(filename):
    title = os.path.split(filename)[1]

    with open(filename) as f:
        nb = nbformat.read(f, 'json')

    for worksheet in nb.worksheets:
        for cell in worksheet.cells:
            if (cell.cell_type == 'heading' or
                (cell.cell_type == 'markdown' and re.match('^#+', cell.source))):
                title = cell.source
                if cell.cell_type != 'heading':
                    title = re.sub('^#+\s*', '', title)  # strip leading '#' symbols
                break

    return title


def make_ipynb_link(name, rawtext, text, lineno, inliner,
                     options={}, content=[]):
    from docutils import nodes, utils
    ipynb_file = os.path.abspath(text)
    ipynb_base = os.path.split(text)[1]
    nb_title = get_notebook_title(ipynb_file)

    ## There is probably a less convoluted way to extract the sphinx_builder...
    ## XXX TODO: And it doesn't work yet anyway...
    #sphinx_builder = inliner.document.settings.env.config.sphinx_builder

    # TODO: The following should be rewritten once nbconvert is
    #       properly integrated into ipython and the API is stable.
    try:
        NBCONVERT = os.environ['NBCONVERT']
    except KeyError:
        print("Please set the environment variable NBCONVERT so that it points "
              "to the location of the script nbconvert.py")
        sys.exit(1)
    nbconvert_path = os.path.dirname(NBCONVERT)
    sys.path.append(nbconvert_path)
    from nbconvert import ConverterHTML

    build_dir = os.path.abspath(os.path.join('_build', 'html'))  # XXX TODO: we should not hardcode the _build/html directory! How to get it from sphinx?

    #print("[DDD] Converting .ipynb file to .html and putting it in target directory")
    # Create the target directories for .ipynb and converted .html files
    for subdir in ['ipynb', 'ipynb_html']:
        if not os.path.exists(os.path.join(build_dir, subdir)):
            os.mkdir(os.path.join(build_dir, subdir))

    # Copy the .ipynb file into the target directory
    shutil.copy(ipynb_file, os.path.join(build_dir, 'ipynb'))

    # Convert the .ipynb file to .html
    html_file = os.path.join('ipynb_html', re.sub('\.ipynb', '.html', ipynb_base))
    c = ConverterHTML(ipynb_file, os.path.join(build_dir, html_file))
    c.render()

    # Create link in the rst tree
    node = nodes.reference(rawtext, nb_title, refuri=html_file, **options)

    nodes = [node]
    sys_msgs = []
    return nodes, sys_msgs


# Setup function to register the extension
def setup(app):
    # app.add_config_value('sphinx_builder',
    #                      app.builder,
    #                      'env')
    app.add_role('ipynb', make_ipynb_link)
