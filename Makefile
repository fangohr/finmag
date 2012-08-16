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

######### Other variables
# Directory where precompiled header files are placed during compilation
export PRECOMPILED_HEADER_DIR = $(PROJECT_DIR)/tmp/$(notdir $(abspath .))-$(BUILD_TAG)-$(BUILD_ID)
# Make sure we do not try to compile native modules when they're imported in python
export DISABLE_PYTHON_MAKE = 1

default:
	@echo 'This makefile is used for CI only; do not use directly.' 
	
ci: test doc

doc: doc-html

doc-html:
	make -C doc generate-doc html

doc-pdf:
	make -C doc generate-doc latexpdf

make-modules:
	make -C $(NATIVE_DIR) all
	
clean:
	make -C $(NATIVE_DIR) clean
	rm -rf test-reports
	$(PURGE_REPO_CMD)

% : %.c

create-dirs:
	mkdir -p test-reports/junit

test: clean make-modules run-unittest-tests run-pytest-tests run-ci-tests

run-unittest-tests : $(addsuffix /__runtests__,$(TEST_ROOTS))

fasttest : make-modules $(addsuffix /__runtests__,$(TEST_ROOTS)) run-ci-tests

%/__runtests__ : create-dirs
	(cd $(dir $@) && NETGENDIR=$(NETGENDIR) PYTHONPATH=$(PYTHON_ROOTS):. python $(RUN_UNIT_TESTS))

run-pytest-tests : create-dirs
	PYTHONPATH=$(PYTHON_ROOTS) py.test src examples --junitxml=$(PROJECT_DIR)/test-reports/junit/TEST_pytest.xml

run-ci-tests :
	make -C $(NATIVE_DIR) run-ci-tests

# This target can be used to print Makefile variables (such as PROJECT_DIR)
# from the command line, for example by saying 'make print-PROJECT_DIR'.
print-%:
	@echo $($*)

.PHONY: ci default make-modules test run-ci-tests run-pytest-tests run-unittest-tests
