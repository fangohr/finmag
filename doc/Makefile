# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = _build
SPHINXWARNINGOPTS ?= -W

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .
# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

# Variables used for converting .ipynb files (= IPython notebooks) to .rst files
NBCONVERT ?= /usr/local/src/nbconvert/nbconvert.py
IPYTHON_NOTEBOOK_SRC_DIR ?= ipython_notebooks_src
IPYTHON_NOTEBOOK_DEST_DIR = ipython_notebooks_dest
IPYTHON_NOTEBOOK_SRC_FILES = $(foreach file,$(wildcard ${IPYTHON_NOTEBOOK_SRC_DIR}/*.ipynb),$(notdir $(basename $(file))))
# If no headers up to this level exist in the .rst file, we add a header
# manually which contains the corresponding .ipynb filename.
IPYTHON_NOTEBOOK_HEADER_LEVEL_THRESHOLD ?= 6

export PYTHONPATH    := ../src:$(PYTHONPATH)

.PHONY: help clean html dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub latex latexpdf text man changes linkcheck doctest gettext

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  dirhtml    to make HTML files named index.html in directories"
	@echo "  singlehtml to make a single large HTML file"
	@echo "  pickle     to make pickle files"
	@echo "  json       to make JSON files"
	@echo "  htmlhelp   to make HTML files and a HTML help project"
	@echo "  qthelp     to make HTML files and a qthelp project"
	@echo "  devhelp    to make HTML files and a Devhelp project"
	@echo "  epub       to make an epub"
	@echo "  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf   to make LaTeX files and run them through pdflatex"
	@echo "  text       to make text files"
	@echo "  man        to make manual pages"
	@echo "  texinfo    to make Texinfo files"
	@echo "  info       to make Texinfo files and run them through makeinfo"
	@echo "  gettext    to make PO message catalogs"
	@echo "  changes    to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck  to check all external links for integrity"
	@echo "  doctest    to run all doctests embedded in the documentation (if enabled)"

clean:
	-rm -rf $(BUILDDIR)/* modules/* ${IPYTHON_NOTEBOOK_DEST_DIR}/* finmagrc_template.txt


htmlraw:
	$(SPHINXBUILD) $(SPHINXWARNINGOPTS) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html

html: generate-doc collect-ipython-notebooks htmlraw prettify-ipython-notebooks
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

dirhtml:
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

singlehtml:
	$(SPHINXBUILD) -b singlehtml $(ALLSPHINXOPTS) $(BUILDDIR)/singlehtml
	@echo
	@echo "Build finished. The HTML page is in $(BUILDDIR)/singlehtml."

pickle:
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(BUILDDIR)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

json:
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) $(BUILDDIR)/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp:
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(BUILDDIR)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDDIR)/htmlhelp."

qthelp:
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) $(BUILDDIR)/qthelp
	@echo
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in $(BUILDDIR)/qthelp, like this:"
	@echo "# qcollectiongenerator $(BUILDDIR)/qthelp/Finmag.qhcp"
	@echo "To view the help file:"
	@echo "# assistant -collectionFile $(BUILDDIR)/qthelp/Finmag.qhc"

devhelp:
	$(SPHINXBUILD) -b devhelp $(ALLSPHINXOPTS) $(BUILDDIR)/devhelp
	@echo
	@echo "Build finished."
	@echo "To view the help file:"
	@echo "# mkdir -p $$HOME/.local/share/devhelp/Finmag"
	@echo "# ln -s $(BUILDDIR)/devhelp $$HOME/.local/share/devhelp/Finmag"
	@echo "# devhelp"

epub:
	$(SPHINXBUILD) -b epub $(ALLSPHINXOPTS) $(BUILDDIR)/epub
	@echo
	@echo "Build finished. The epub file is in $(BUILDDIR)/epub."

latex:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make' in that directory to run these through (pdf)latex" \
	      "(use \`make latexpdf' here to do that automatically)."

latexpdf:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

text:
	$(SPHINXBUILD) -b text $(ALLSPHINXOPTS) $(BUILDDIR)/text
	@echo
	@echo "Build finished. The text files are in $(BUILDDIR)/text."

man:
	$(SPHINXBUILD) -b man $(ALLSPHINXOPTS) $(BUILDDIR)/man
	@echo
	@echo "Build finished. The manual pages are in $(BUILDDIR)/man."

texinfo:
	$(SPHINXBUILD) -b texinfo $(ALLSPHINXOPTS) $(BUILDDIR)/texinfo
	@echo
	@echo "Build finished. The Texinfo files are in $(BUILDDIR)/texinfo."
	@echo "Run \`make' in that directory to run these through makeinfo" \
	      "(use \`make info' here to do that automatically)."

info:
	$(SPHINXBUILD) -b texinfo $(ALLSPHINXOPTS) $(BUILDDIR)/texinfo
	@echo "Running Texinfo files through makeinfo..."
	make -C $(BUILDDIR)/texinfo info
	@echo "makeinfo finished; the Info files are in $(BUILDDIR)/texinfo."

gettext:
	$(SPHINXBUILD) -b gettext $(I18NSPHINXOPTS) $(BUILDDIR)/locale
	@echo
	@echo "Build finished. The message catalogs are in $(BUILDDIR)/locale."

changes:
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(BUILDDIR)/changes
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest:
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."

#next target added to get Jenkins to run generate_doc.py before building the documentation.
generate-doc: finmagrc_template.txt examples-for-doc timing
	python generate_doc.py

finmagrc_template.txt:
	python -c "import finmag.util.configuration as c; c.write_finmagrc_template_to_file('finmagrc_template.txt')"

# The absolute path for the doc directory
DOC_DIR = $(abspath .)
# Paths that have to be added to PYTHONPATH prior to running the tests from here (for jenkins)
PYTHON_ROOTS = $(DOC_DIR)/../src

# Benchmark nmag example #2.
timing:
	make -C ../examples/exchange_demag/timings/

# Collect all .ipynb files in IPYTHON_NOTEBOOK_SRC_DIR and convert
# them to .rst files in IPYTHON_NOTEBOOK_DEST_DIR, which are then
# included in the documentation
collect-ipython-notebooks: ${IPYTHON_NOTEBOOK_DEST_DIR}
	@echo "IPYTHON_NOTEBOOK_SRC_DIR: ${IPYTHON_NOTEBOOK_DEST_DIR}"
	@command -v ${NBCONVERT} >/dev/null 2>&1 || { echo >&2 "Not collecting ipython notebooks because the variable NBCONVERT does not point to a valid nbconvert.py script (value: ${NBCONVERT})"; exit 1; }
	-rm -rf ${IPYTHON_NOTEBOOK_DEST_DIR}/*
	@echo "Converting IPython notebooks in directory '${IPYTHON_NOTEBOOK_SRC_DIR}' to .rst format..."
	@for FILE in ${IPYTHON_NOTEBOOK_SRC_FILES} ; do \
	    echo "    Converting file: $$FILE"; \
	    ${NBCONVERT} --format=rst ${IPYTHON_NOTEBOOK_SRC_DIR}/$$FILE.ipynb ${IPYTHON_NOTEBOOK_DEST_DIR}/$$FILE.rst && \
	    python add_heading_if_needed.py ${IPYTHON_NOTEBOOK_DEST_DIR}/$$FILE.rst ${IPYTHON_NOTEBOOK_HEADER_LEVEL_THRESHOLD}; \
	done

${IPYTHON_NOTEBOOK_DEST_DIR}:
	install -d ${IPYTHON_NOTEBOOK_DEST_DIR}

# The following target overwrites the existing .html files in ipython_notebooks_dest/
# (which were produced via the chain .ipynb -> .rst -> .html) with ones that are
# directly generated from .ipynb using nbconvert. These look much nicer and closer to
# the original, and are thus desirable. Note that this is a bad hack because sphinx
# we rely on a lot of implicit assumptions and sphinx doesn't know that the files were
# changed, but this should not cause any problems for now. We can revisit it in a bit
# and clean it up.  -- Max, 14.12.2012
prettify-ipython-notebooks:
	@echo "Replacing notebook html files with nicer ones"
	@for FILE in ${IPYTHON_NOTEBOOK_SRC_FILES} ; do \
	    echo "Replacing file: $$FILE"; \
	    ${NBCONVERT} --format=html ${IPYTHON_NOTEBOOK_SRC_DIR}/$$FILE.ipynb _build/html/${IPYTHON_NOTEBOOK_DEST_DIR}/$$FILE.html; \
	done

#this target can be run to execute py.test ../examples to get figure
#files ready. This will happen automatically when Jenkins runs, as
#all tests in finmag/src and finmag/examples are run before the 

#list of files we need for documentation and which should trigger running of
#the right files in ../examples to create them.
examples-for-doc: ../examples/macrospin/alpha-1-00.png ../examples/macrospin/alpha-0-50.png ../examples/macrospin/alpha-0-10.png \
	../examples/macrospin/alpha-0-02.png ../examples/time-dependent-applied-field/fittedresults.txt \
	../examples/time-dependent-applied-field/fit.png ../examples/time-dependent-applied-field/results.png \
	../examples/exchange_demag/exchange_demag.png ../examples/exchange_demag/exchange_energy.png \
	../examples/exchange_demag/exchange_density.png ../examples/exchange_demag/demag_density.png \
	../examples/exchange_demag/demag_energy.png ../examples/demag/stddev_loglog.png ../examples/demag/xvalues.png \
        ../examples/demag/linsolveparams.rst ../examples/demag/xvaluesgcr.png ../examples/demag/bemtimings.png \
        ../examples/demag/solvetimings.png \
	../examples/demag/errnorm_loglog.png ../examples/demag/results_field.txt ../examples/demag/demagenergies.txt \
	../src/finmag/tests/comparison/exchange/table.rst ../src/finmag/tests/comparison/anisotropy/table.rst \
	../src/finmag/tests/comparison/demag/table.rst ../examples/exchange_demag/simple1D.png \
	../examples/exchange_demag/finmag_exch_density.npy ../examples/exchange_demag/finmag_demag_density.npy \
	../examples/spatially-varying-anisotropy/profile.png \
	../examples/exchange_demag/timings/results.rst ../examples/precession/precession.png \
	../examples/nmag_example_2/agreement_averages.png

../examples/nmag_example_2/agreement_averages.png:
	PYTHONPATH=$(PYTHON_ROOTS) $(MAKE) -C ../examples/nmag_example_2 all

../examples/macrospin/alpha-1-00.png: ../examples/macrospin/test_macrospin.py
	cd ../examples/macrospin/ && PYTHONPATH=$(PYTHON_ROOTS) python test_macrospin.py

../examples/macrospin/alpha-0-50.png: ../examples/macrospin/test_macrospin.py
	cd ../examples/macrospin/ && PYTHONPATH=$(PYTHON_ROOTS) python test_macrospin.py

../examples/macrospin/alpha-0-10.png: ../examples/macrospin/test_macrospin.py
	cd ../examples/macrospin/ && PYTHONPATH=$(PYTHON_ROOTS) python test_macrospin.py

../examples/macrospin/alpha-0-02.png: ../examples/macrospin/test_macrospin.py
	cd ../examples/macrospin/ && PYTHONPATH=$(PYTHON_ROOTS) python test_macrospin.py

../examples/time-dependent-applied-field/fittedresults.txt ../examples/time-dependent-applied-field/fit.png ../examples/time-dependent-applied-field/results.png: ../examples/time-dependent-applied-field/test_appfield.py
	cd ../examples/time-dependent-applied-field && PYTHONPATH=$(PYTHON_ROOTS) python test_appfield.py

../examples/exchange_demag/exchange_demag.png ../examples/exchange_demag/exchange_energy.png ../examples/exchange_demag/demag_energy.png ../examples/exchange_demag/exchange_density.png ../examples/exchange_demag/demag_density.png: ../examples/exchange_demag/test_exchange_demag.py
	cd ../examples/exchange_demag/ && PYTHONPATH=$(PYTHON_ROOTS) python test_exchange_demag.py

../examples/demag/stddev_loglog.png ../examples/demag/xvalues.png ../examples/demag/xvaluesgcr.png ../examples/demag/errnorm_loglog.png ../examples/demag/linsolveparams.rst ../examples/demag/krylovitr.png ../examples/demag/bemtimings.png ../examples/demag/solvetimings: ../examples/demag/short_nmag_comp_tests2.py
	cd ../examples/demag/ && PYTHONPATH=$(PYTHON_ROOTS) python short_nmag_comp_tests2.py

../examples/exchange_demag/timings/results.rst: ../examples/exchange_demag/timings/run_finmag.py
	cd ../examples/exchange_demag/timings/ && PYTHONPATH=$(PYTHON_ROOTS) python run_finmag.py

../examples/demag/results_field.txt: ../examples/demag/test_field.py
	cd ../examples/demag/ && PYTHONPATH=$(PYTHON_ROOTS) python test_field.py > results_field.txt

../examples/demag/demagenergies.txt: ../examples/demag/test_energy.py
	cd ../examples/demag/ && PYTHONPATH=$(PYTHON_ROOTS) python test_energy.py

../src/finmag/tests/comparison/exchange/table.rst:
	cd ../src/finmag/tests/comparison/exchange/ && PYTHONPATH=$(PYTHON_ROOTS) python test_exchange_field.py

../src/finmag/tests/comparison/anisotropy/table.rst:
	cd ../src/finmag/tests/comparison/anisotropy/ && PYTHONPATH=$(PYTHON_ROOTS) python anis_all.py

../src/finmag/tests/comparison/demag/table.rst:
	cd ../src/finmag/tests/comparison/demag/ && PYTHONPATH=$(PYTHON_ROOTS) python test_demag_field.py

../examples/exchange_demag/simple1D.png:
	cd ../examples/exchange_demag/ && PYTHONPATH=$(PYTHON_ROOTS) python simple_1D_finmag.py

../examples/exchange_demag/finmag_exch_density.npy ../examples/exchange_demag/finmag_demag_density.npy:
	cd ../examples/exchange_demag/ && make nmag

../examples/spatially-varying-anisotropy/profile.png:
	cd ../examples/spatially-varying-anisotropy && PYTHONPATH=$(PYTHON_ROOTS) python run.py

../examples/std_prob_3/table.rst:
	cd ../examples/std_prob_3 && PYTHONPATH=$(PYTHON_ROOTS) python run.py

../examples/precession/precession.png:
	cd ../examples/precession && PYTHONPATH=$(PYTHON_ROOTS) python run.py

# This target can be used to print Makefile variables (such as PROJECT_DIR)
# from the command line, for example by saying 'make print-PROJECT_DIR'.
print-%:
	@echo $($*)
