About the documentation
=======================


Required libraries
------------------

You will probably need Sphinx version 1.1.2 (more recent that what ships with Ubuntu 11.10) which can be installed using::

  $ easy_install -U Sphinx

To get syntax highlighting, you also need Pygments (the Ubuntu package 'python-pygments' does the job) or use::

  $ easy_install Pygments


To build and view documentation
-------------------------------

1. Navigate to finmag/doc/
2. ``$ python generate_doc.py``
3. ``$ make html``
4. Open ``_build/html/index.html`` in your favourite browser

Other formats can be created, for example pdf using ``make latexpdf`` (the output is in ``_build/latexpdf/Finmag.pdf``).


Writing documentation
---------------------

We are using Sphinx (http://sphinx.pocoo.org/) to generate documentation. Basic use includes simple inline markup, using mathematics in LaTeX syntax, and code snippets.

When using the math environment, use :math:`E = mc^2` for inline mathematics (corresponding to $E=mc^2$ in LaTeX), and

.. math::

    E = mc^2

for equation style mathematics (correspoding to \[ ... \]). Note that we have to use double backslash (\\nabla instead of \nabla) or else some of the interpreters think it indicates a new line, tabulator, etc. 

To write python code, simply use

.. code-block:: python

    >>> print 'Hello, World!'

For demonstration, check out finmag.sim.anisotropy and finmag.sim.exchange.

For more information, see http://fenicsproject.org/contributing/styleguide_doc.html#styleguide-documentation 

When you have documented your code, and want to add it to the html files, please do as follows:

*    Open /finmag/doc/modules.txt
*    Add to the list of modules a line containing the name you want in the webpage header (this can be several words separated by spaces) and the path to the module (this must NOT include spaces or /). For example, if you want to add /finmag/sim/anisotropy.py, add the line::

       Anisotropy finmag.sim.anisotropy

     to modules.txt.

and compile again.


Important
---------

When documenting a class structure, Sphinx does not include docstrings from __init__, so please put these in the first line in the class instead. I.e. do

.. code-block:: python

   class Foo:
       """
       Docstring is here.
       """
   
       def __init__(self, *args):
   
   
instead of

.. code-block:: python   

   class Bar:
       def __init__(self, *args):
           """
           Docstring should not be here.
           """

Code-blocks
-----------

Need to be preceeded by::

  .. code-block:: python

and the code block has to be indented.

There are additional options for ``.. code-block::``, and also a related command ``.. literalinclude::``. Both are explained at http://sphinx.pocoo.org/markup/code.html and provide the possibility

* to include only particular lines
* only a particular class or Python object from a given file (funky if it actually works!)
* display of line numbers
* and more.




This document as raw source
---------------------------

Here we include the source for this page

.. literalinclude:: thisdocumentation.rst


----------------

# HOWTO generate documentation for Finmag.
# Written by Anders E. Johansen 23/2/2012
