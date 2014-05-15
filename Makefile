# FinMag - Copyright (C) 2012, 2013, 2014 University of Southampton
# Contact: Hans Fangohr <h.fangohr@soton.ac.uk>
# Do not distribute.
#
# This Makefile is used by our continuous-integration server
# to run the tests and to build the documentation 
#

default:
	@echo 'This makefile is used for CI only; do not use directly.' 

PYTHON ?= python
PURGE_REPO_CMD ?= hg purge --all
NETGENDIR ?= /usr/share/netgen/
PROJECT_DIR = $(abspath .)
NATIVE_DIR = native
PYTHON_ROOTS = $(PROJECT_DIR)/src
FINMAG_REPO ?= ssh://hg@bitbucket.org/fangohr/finmag
FINMAG_BINARY_DEST ?= $(HOME)/finmag_binary_version_of_last_successful_build
FINMAG_BINARY_LICENSE_FILE ?= $(HOME)/License-LicenseRequest_FinmagJenkins
DIST_WRAPPER_DIR ?= $(HOME)/finmag-dist
DIST_WRAPPER_OPTIONS ?= --skip-tests --finmag-repo=$(FINMAG_REPO) --destdir=$(FINMAG_BINARY_DEST)

export PRECOMPILED_HEADER_DIR = $(PROJECT_DIR)/tmp/$(notdir $(abspath .))-$(BUILD_TAG)-$(BUILD_ID)
export DISABLE_PYTHON_MAKE = 1  # to only build native modules once per session

print-debugging-info: print-PROJECT_DIR print-PYTHON_ROOTS print-NETGENDIR
	@echo

# display variables defined in this Makefile
# example: `make print-PROJECT_DIR`
print-%:
	@echo [DDD] Makefile $*=\`$($*)\`

purge: clean
	@echo "Removing all untracked files from repository."
	$(PURGE_REPO_CMD)

clean:
	make -C src/ cleansrc
	make -C $(NATIVE_DIR) clean
	rm -rf test-reports

##################################
# BUILDING THE NATIVE EXTENSIONS #
##################################

make-modules:
	make -C $(NATIVE_DIR) all

% : %.c

##############################
# BUILDING THE DOCUMENTATION #
##############################

doc: make-modules doc-html doc-latexpdf doc-singlehtml

doc-clean:
	make -C doc clean

# generate documentation in html format without re-building the examples
# good for debugging but needs to the examples to have been built before
doc-html-nobuildexamples:
	SPHINXWARNINGOPTS= make -C doc htmlraw

# generate documentation in any of the supported formats in doc/Makefile
# examples: `make doc-html`, `make doc-singlehtml`, `make doc-pdf` 
doc-%:
	make -C doc generate-doc $*

#####################
# RUNNING THE TESTS #
#####################

# py.test options
# example: `-sx` to disable capturing of STDOUT and exit on first error
TEST_OPTIONS ?=

create-dirs:
	mkdir -p test-reports/junit

test: clean create-dirs make-modules tests tests-native tests-notebooks

# run all Python tests
test-python: create-dirs make-modules
	PYTHONPATH=$(PYTHON_ROOTS) py.test $(TEST_OPTIONS) \
		--junitxml=$(PROJECT_DIR)/test-reports/junit/TEST_pytest.xml

# exclude tests marked as slow
test-fast: create-dirs make-modules
	PYTHONPATH=$(PYTHON_ROOTS) py.test $(TEST_OPTIONS) -m "not slow" \
		--junitxml=$(PROJECT_DIR)/test-reports/junit/TEST_pytest.xml

# only run tests marked as slow
test-slow: create-dirs make-modules
	PYTHONPATH=$(PYTHON_ROOTS) py.test $(TEST_OPTIONS) -m "slow" \
		--junitxml=$(PROJECT_DIR)/test-reports/junit/TEST_pytest.xml

# run both fast and slow tests
tests: test-fast test-slow

# run tests on the native extensions
test-native: make-modules
	make -C $(NATIVE_DIR) run-ci-tests

# try to reproduce the ipython notebooks
test-notebooks: create-dirs make-modules print-debugging-info
	PYTHONPATH=$(PYTHON_ROOTS) echo "[DDD] PYTHONPATH now: ${PYTHONPATH}" && py.test $(TEST_OPTIONS) bin/reproduce_ipython_notebooks.py --junitxml=$(PROJECT_DIR)/test-reports/junit/TEST_pytest.xml

.PHONY: default make-modules create-dirs doc test test-python test-fast test-slow test-native test-notebooks
