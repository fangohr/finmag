# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

#######################################
# This makefile compiles the finmag extension module.
#
# Adjustable options are below.
#
all : parallel
# Read the automatically generated dependency files
-include $(shell find . -name \*.d)
#######################################
#  Configurable options
#
# The directory where to place the extension modules
MODULE_DIR = ../src/finmag/native
# The list of extension modules to build
MODULES = sundials.so llg.so llb.so fast_sum_lib.so treecode_bem.so
# The directory to store the precompiled header file.
PRECOMPILED_HEADER_DIR ?= /tmp/$(USER)/finmag-pch/$(abspath .)
# Name of the binary that runs all unit tests
UNIT_TEST_BINARY=$(MODULE_DIR)/run_ci_tests.bin
# Directory for CI unit test results
UNIT_TEST_RESULTS_DIR=../test-reports/boost
UNIT_TEST_RESULTS_FILE=$(UNIT_TEST_RESULTS_DIR)/run_ci_tests_output.xml
# Explicit dependencies to tell how the module files are linked
SUNDIALS_NVECTOR_SO=-Wl,-rpath,\$$ORIGIN -L$(abspath ../src/finmag/native/) -lnvector_serial_custom_malloc

$(MODULE_DIR)/sundials.so : src/sundials/py_sundials_module.o src/util/np_array.o src/sundials/numpy_malloc.o
sundials_LDFLAGS =-lsundials_cvode -llapack $(SUNDIALS_NVECTOR_SO) -lboost_thread

$(MODULE_DIR)/llg.so : src/llg/py_llg_module.o src/util/np_array.o src/util/swig_dolfin.o src/llg/bem.o src/llg/llg.o src/llg/heun.o
llg_LDFLAGS = -L$(HOME)/Work/FEniCS/lib -ldolfin

$(MODULE_DIR)/llb.so : src/llb/py_llb_module.o src/util/np_array.o src/llb/material.o src/llb/llb.o src/llb/sllg.o src/llb/sllg_stt.o src/llb/mt19937.o 

$(MODULE_DIR)/fast_sum_lib.so:
	cd src/fast_sum_lib && python setup.py build_ext --inplace
	install src/fast_sum_lib/fast_sum_lib.so ../src/finmag/native

$(MODULE_DIR)/treecode_bem.so: src/treecode_bem/treecode_bem_I.c  src/treecode_bem/treecode_bem_II.c src/treecode_bem/common.c src/treecode_bem/treecode_bem_lib.pyx
	cd src/treecode_bem && python setup.py build_ext --inplace
	install src/treecode_bem/treecode_bem.so ../src/finmag/native

$(UNIT_TEST_BINARY) : tests/run_ci_tests.o src/util/np_array.o src/sundials/tests/sundials_tests.o src/sundials/numpy_malloc.o 
run_ci_tests_LDFLAGS=-lboost_prg_exec_monitor /usr/lib/libboost_unit_test_framework.a $(SUNDIALS_NVECTOR_SO) -L$(HOME)/Work/FEniCS/lib -ldolfin

#######################################
# Other variables
#
# The list of output binary files to link
BUILD_TARGETS = $(addprefix $(MODULE_DIR)/,$(MODULES)) $(UNIT_TEST_BINARY) add_version
# The list of source files to be compiled
SOURCES = $(shell find src tests -name \*.cc)
# The list of object files to be compiled
OBJECT_FILES = $(patsubst %.cc,%.o,$(SOURCES))

# The name of the precompiled header file.
PRECOMPILED_HEADER_FILE=$(PRECOMPILED_HEADER_DIR)/finmag_includes.h.gch
CFLAGS_USE_PCH = -include $(PRECOMPILED_HEADER_DIR)/finmag_includes.h

######################
# Compilation flags
#
# N.B.: For building the Finmag binary distribution: replace
#       '-march=native' with '-march=x86-64 -mtune=generic' in
#       CFLAGS_OPT in order to have the most portable compiler flags
#       (which is important if the machine where the Finmag binary is
#       built is different from the one where it is going to be used
#       the machine where it is going to be used). We have this
#       setting in the finmag-dist repository, so this comment is just
#       to document these flags.
CFLAGS_OPT ?= -O3 -funroll-loops -mfpmath=sse -march=native
CFLAGS_PYTHON=-I/usr/include/python2.7 -fPIC -lboost_python -lpython2.7
# -rdynamic keeps private symbols in .so files, providing more detailed stack traces
# -MMD -MP generates .d dependency files
CFLAGS_OTHER=-g -rdynamic -fopenmp -MMD -MP -std=c++0x
# Treat all warnings as errors
CFLAGS_WARN=-Wall -Werror
# Add the source code directory to the include path
CFLAGS_INCLUDE=-I$(abspath src)
CFLAGS_NOPCH=$(CFLAGS_OPT) $(CFLAGS_PYTHON) $(CFLAGS_OTHER) $(CFLAGS_WARN) $(CFLAGS_INCLUDE)
CFLAGS=$(CFLAGS_NOPCH) $(CFLAGS_USE_PCH)

# The number of make jobs to run in parallel
N_SIMULTANEOUS_JOBS:=$(shell grep -ci processor /proc/cpuinfo)

FINMAGDIST_REVISION ?= tip

REVISION_NR = $(shell hg id -n -r ${FINMAGDIST_REVISION})
REVISION_ID = $(shell hg --debug id -i -r ${FINMAGDIST_REVISION})

# TODO: This is probably not the best place for this target, but at
#       least it ensures that the version is re-generated whenever the
#       native code is compiled, so we are always up to date.
add_version:
	@echo "FINMAG_VERSION: ${REVISION_NR}:${REVISION_ID}"
	echo '__version__ = "${REVISION_NR}:${REVISION_ID}"' > ../src/finmag/__version__.py

# Builds the whole project
parallel:
	@echo ------------------
	@echo Building object files: $(OBJECT_FILES)
	@echo ------------------
	$(MAKE) -j $(N_SIMULTANEOUS_JOBS) object-files subprojects
	@echo ------------------
	@echo Linking: $(BUILD_TARGETS)
	@echo ------------------
	$(MAKE) -j $(N_SIMULTANEOUS_JOBS) build-targets

# Links the object files and creates the binaries
build-targets : $(BUILD_TARGETS)

# Compiles the source files into object files
object-files : $(OBJECT_FILES)

# Builds subprojects that have separate makefiles
subprojects : nvec_serial

# Build the custom nvector_serial library
nvec_serial :
	@$(MAKE) -C src/sundials/nvec_serial

# Builds an object file from a .cc file
%.o : %.cc $(PRECOMPILED_HEADER_FILE) Makefile
	@echo Building object file: $@
	@$(CXX) -c $< $(CFLAGS) $(MODULE_FLAGS) -o $@

# Creates the precompiled header file
pch: $(PRECOMPILED_HEADER_FILE)
$(PRECOMPILED_HEADER_FILE) : src/finmag_includes.h Makefile
	@echo Building precompiled header file: $@
	@mkdir -p $(PRECOMPILED_HEADER_DIR)
	@$(CXX) $(CFLAGS_NOPCH) -o $@ -c $<

# Runs Boost.Test unit tests in native code
run-ci-tests:
	$(MAKE) -j $(NUM_PROCS) $(UNIT_TEST_BINARY)
	mkdir -p $(UNIT_TEST_RESULTS_DIR)
	./$(UNIT_TEST_BINARY) --output_format=XML --report_level=no --log_level=test_suite > $(UNIT_TEST_RESULTS_FILE)

# Removes all temporary files
clean:
	rm -f $(BUILD_TARGETS) $(PRECOMPILED_HEADER_FILE)
	find . \( -name \*.o -o -name \*.d -o -name \*.o.nm -o -name \*.gen.cc -o -name \*.gen.h -o -name \*.so \) -exec rm {} +

# Runs the native unit tests
test:
	$(MAKE) -j $(NUM_PROCS) parallel
	./$(UNIT_TEST_BINARY)

.PHONY: clean pch all object-files build-targets parallel run-ci-tests subprojects test

# Use .SECONDEXPANSION to use the per-library linking flags *_LDFLAGS
.SECONDEXPANSION:

# Links a Python extension module
%.so :
	@echo Building shared library: $@
	$(CXX) $^ $(CFLAGS) $(MODULE_FLAGS) $($(patsubst %.so,%,$(notdir $@))_LDFLAGS) -shared -o $@

# Links a binary executable
%.bin :
	@echo Building binary: $@
	$(CXX) $^ $(CFLAGS) $(MODULE_FLAGS) $($(patsubst %.bin,%,$(notdir $@))_LDFLAGS) -o $@

# This target can be used to print Makefile variables (such as CFLAGS_OPT)
# from the command line, for example by saying 'make print-CFLAGS_OPT'.
print-%:
	@echo $($*)
