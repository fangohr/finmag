# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

#######################################
# Currently, this makefile is only used to perform the Jenkins build
# 
# Jenkins will call the 'ci' (for Continuous Integration) target.
#
# 

# The directory containing ng.tcl
NETGENDIR ?= /usr/share/netgen/
PYTHON ?= python

# The command to purge all untracked files in the repository.
# If you use hg-git to inter-operate with the hg repository,
# you may want to use:
#
#    PURGE_REPO_CMD = git clean -f -d
#
PURGE_REPO_CMD ?= hg purge --all

# The absolute path for the project directory
PROJECT_DIR = $(abspath .)

######### Project-specific variables
# Paths that have to be added to PYTHONPATH prior to running the tests
PYTHON_ROOTS = $(PROJECT_DIR)/src
# This script runs all unit tests found in the current directory
RUN_UNIT_TESTS = $(PROJECT_DIR)/src/finmag/util/run_ci_tests.py
# The directory that contains native code
NATIVE_DIR = native
# The list of directories that contain unittest unit tests
TEST_ROOTS = src
# Set TEST_OPTIONS to e.g. '-sxv' to disable capturing of standard output, exit instantly on the first error, and increase verbosity.
TEST_OPTIONS ?=

######### Other variables
# Directory where precompiled header files are placed during compilation
export PRECOMPILED_HEADER_DIR = $(PROJECT_DIR)/tmp/$(notdir $(abspath .))-$(BUILD_TAG)-$(BUILD_ID)
# Make sure we do not try to compile native modules when they're imported in python
export DISABLE_PYTHON_MAKE = 1
# Where the tarball containing the binary version of the latest succesful build should be placed and extracted
FINMAG_BINARY_DEST ?= $(HOME)/finmag_binary_version_of_last_successful_build
# The repo to clone when building the binary tarball
FINMAG_REPO ?= ssh://hg@bitbucket.org/fangohr/finmag
FINMAG_BINARY_LICENSE_FILE ?= $(HOME)/License-LicenseRequest_FinmagJenkins
# The directory where the script dist-wrapper.py lives
DIST_WRAPPER_DIR ?= $(HOME)/finmag-dist
# Command line options passed to dist-wrapper.py
DIST_WRAPPER_OPTIONS ?= --skip-tests --finmag-repo=$(FINMAG_REPO) --destdir=$(FINMAG_BINARY_DEST)

default:
	@echo 'This makefile is used for CI only; do not use directly.' 

ci: purge test doc

doc: doc-html doc-pdf doc-singlehtml

doc-html:
	make -C doc generate-doc html

doc-singlehtml:
	make -C doc generate-doc singlehtml

doc-pdf:
	make -C doc generate-doc latexpdf

doc-clean:
	make -C doc clean

# The following is useful for quick debugging as it doesn't rebuild the examples.
# However, currently it will only work if a complete run of 'make doc-html' was
# successfully performed beforehand.
doc-html-nobuildexamples:
	SPHINXWARNINGOPTS= make -C doc htmlraw

make-modules:
	make -C $(NATIVE_DIR) all

update-jenkins-binary-version:
ifeq "$(shell hostname)" "summer"
	@echo "Removing existing binary installation and tarball(s) in directory ${FINMAG_BINARY_DEST}"
#	-rm -f ${FINMAG_BINARY_DEST}/FinMag*.tar.bz2
	-rm -rf ${FINMAG_BINARY_DEST}/finmag
	@echo "Installing latest binary version in directory ${FINMAG_BINARY_DEST}"
	cd $(DIST_WRAPPER_DIR) && $(PYTHON) dist-wrapper.py $(DIST_WRAPPER_OPTIONS)
	tar -C ${FINMAG_BINARY_DEST} -xjf ${FINMAG_BINARY_DEST}/FinMag*.tar.bz2
	install ${FINMAG_BINARY_LICENSE_FILE} ${FINMAG_BINARY_DEST}/finmag
else
	@echo "The Makefile target $@ only makes sense"
	@echo "to execute on summer.kk.soton.ac.uk as part of the CI process."
	@echo "Quitting since we appear to be on a different machine."
endif

clean:
	make -C src/ cleansrc
	make -C $(NATIVE_DIR) clean
	rm -rf test-reports

purge: clean
	@echo "Removing all untracked files from repository."
	$(PURGE_REPO_CMD)

% : %.c

create-dirs:
	mkdir -p test-reports/junit

test: clean print-debugging-info make-modules run-unittest-tests run-pytest-tests run-ci-tests

print-debugging-info:
	@echo "[DDD] Makefile NETGENDIR: ${NETGENDIR}"

run-unittest-tests : $(addsuffix /__runtests__,$(TEST_ROOTS))

fasttest : make-modules $(addsuffix /__runtests__,$(TEST_ROOTS)) run-ci-tests

%/__runtests__ : create-dirs
	(cd $(dir $@) && NETGENDIR=$(NETGENDIR) PYTHONPATH=$(PYTHON_ROOTS):. python $(RUN_UNIT_TESTS))

run-pytest-tests : create-dirs
	PYTHONPATH=$(PYTHON_ROOTS) py.test $(TEST_OPTIONS) src examples --junitxml=$(PROJECT_DIR)/test-reports/junit/TEST_pytest.xml

run-ci-tests :
	make -C $(NATIVE_DIR) run-ci-tests

# This target can be used to print Makefile variables (such as PROJECT_DIR)
# from the command line, for example by saying 'make print-PROJECT_DIR'.
print-%:
	@echo $($*)

.PHONY: ci default make-modules test run-ci-tests run-pytest-tests run-unittest-tests
