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
# Python executable used for native builds.
PYTHON ?= python
PYTHON_CONFIG ?= $(PYTHON)-config
PYTHON_BOOST_TAG := py$(shell $(PYTHON) -c 'import sys; print("{}{}".format(sys.version_info[0], sys.version_info[1]))')
PYTHON_CONFIG_EXISTS := $(shell command -v $(PYTHON_CONFIG) >/dev/null 2>&1 && echo 1 || echo 0)
PYTHON_PREFIX := $(shell $(PYTHON) -c 'import sys; print(sys.prefix)')
NUMPY_INCLUDE_DIR := $(shell $(PYTHON) -c 'import numpy; print(numpy.get_include())')
DETECTED_BOOST_PYTHON_LIB := $(shell find "$(PYTHON_PREFIX)/lib" -maxdepth 1 -type f -name 'libboost_python*.so*' | sed -n '1s|.*/lib||;1s|\.so.*||p')
# Managed environments often ship a different Boost.Python soname from the
# old Debian-based `boost_python-pyXY` pattern, so detect it from the active
# prefix first and only fall back to the historical name. [Codex GPT-5.4]
ifeq ($(strip $(DETECTED_BOOST_PYTHON_LIB)),)
BOOST_PYTHON_LIB ?= boost_python-$(PYTHON_BOOST_TAG)
else
BOOST_PYTHON_LIB ?= $(DETECTED_BOOST_PYTHON_LIB)
endif

# Fenics 2016.2 broke the building by changing where ufc.h is located.
UFC_INC=$(shell if [ -f /usr/include/ufc.h ]; then echo -I/usr/include; else echo -I/usr/lib/python2.7/dist-packages/ffc/backends/ufc/; fi)
# The directory where to place the extension modules
MODULE_DIR = ../src/finmag/native
# The first pixi probe targets import/construction and SciPy-backed stepping,
# so build only the native paths needed for the current pixi milestone and
# fold Sundials back in as that port becomes viable. [Codex GPT-5.4]
# [Codex GPT-5.4]
ifneq ($(findstring /.pixi/,$(PYTHON_PREFIX)),)
MODULES = sundials.so llg.so llb.so
else
# The list of extension modules to build
MODULES = sundials.so llg.so llb.so fast_sum_lib.so neb.so #cvode_petsc.so
endif
# The directory to store the precompiled header file.
PRECOMPILED_HEADER_DIR ?= /tmp/$(USER)/finmag-pch/$(abspath .)
# Name of the binary that runs all unit tests
UNIT_TEST_BINARY=$(MODULE_DIR)/run_ci_tests.bin
# Directory for CI unit test results
UNIT_TEST_RESULTS_DIR=../test-reports/boost
UNIT_TEST_RESULTS_FILE=$(UNIT_TEST_RESULTS_DIR)/run_ci_tests_output.xml
# Explicit dependencies to tell how the module files are linked
SUNDIALS_NVECTOR_SO=-lsundials_nvecserial

$(MODULE_DIR)/sundials.so : src/sundials/py_sundials_module.o src/util/np_array.o src/sundials/numpy_malloc.o
ifneq ($(findstring /.pixi/,$(PYTHON_PREFIX)),)
# Modern SUNDIALS splits CVODE, linear solvers, and matrix backends into
# separate shared libraries, so link the wrapper against the actively used
# SUNLINSOL/SUNMATRIX components as well. [Codex GPT-5.4]
sundials_LDFLAGS =-lsundials_cvode -lsundials_core -lsundials_sunlinsoldense -lsundials_sunlinsollapackdense -lsundials_sunlinsolband -lsundials_sunlinsollapackband -lsundials_sunlinsolspgmr -lsundials_sunlinsolspbcgs -lsundials_sunlinsolsptfqmr -lsundials_sunmatrixdense -lsundials_sunmatrixband -llapack $(SUNDIALS_NVECTOR_SO) -lboost_thread 
else
sundials_LDFLAGS =-lsundials_cvode -llapack $(SUNDIALS_NVECTOR_SO) -lboost_thread
endif

#SUNDIALS_CUSTOM=-I$(SUNDIALS_PATH)/include -Wl,-rpath, -Wl,-L$(SUNDIALS_PATH)/lib
#$(MODULE_DIR)/sundials_p.so : src/sundials_p/py_cvode_module.o src/util/np_array.o src/util/swig_dolfin.o
#sundials_p_LDFLAGS = $(SUNDIALS_CUSTOM) -lsundials_cvode -lsundials_nvecserial -lsundials_nvecparallel -llapack -ldolfin -lboost_thread

$(MODULE_DIR)/llg.so : src/llg/py_llg_module.o src/util/np_array.o src/util/swig_dolfin.o src/llg/bem.o src/llg/llg.o src/llg/heun.o src/llg/energy.o
llg_LDFLAGS = -L$(INSTALL_PATH)/lib -ldolfin

$(MODULE_DIR)/llb.so : src/llb/py_llb_module.o src/util/np_array.o src/llb/material.o src/llb/llb.o src/llb/sllg.o src/llb/sllg_stt.o src/llb/mt19937.o 

$(MODULE_DIR)/neb.so : src/neb/py_neb_module.o src/util/np_array.o src/neb/helper.o

#$(MODULE_DIR)/fast_sum_lib.so:
#	cd src/fast_sum_lib && $(PYTHON) setup.py build_ext --inplace
#	install src/fast_sum_lib/fast_sum_lib.so ../src/finmag/native

$(MODULE_DIR)/treecode_bem.so: src/treecode_bem/treecode_bem_I.c  src/treecode_bem/treecode_bem_II.c src/treecode_bem/common.c src/treecode_bem/bem_pbc.c src/treecode_bem/treecode_bem_lib.pyx 
	cd src/treecode_bem && $(PYTHON) setup.py build_ext --inplace
	install src/treecode_bem/treecode_bem*.so ../src/finmag/native

#$(MODULE_DIR)/cvode_petsc.so: src/cvode/cvode_petsc.pxd  src/cvode/cvode_petsc.pyx src/cvode/llg.c src/cvode/llg.h src/cvode/llg_petsc.pyx
#	cd src/cvode && $(PYTHON) setup.py build_ext --inplace
#	install src/cvode/cvode_petsc.so ../src/finmag/native
#	install src/cvode/llg_petsc.so ../src/finmag/native

$(UNIT_TEST_BINARY): tests/run_ci_tests.o src/util/np_array.o src/sundials/tests/sundials_tests.o src/sundials/numpy_malloc.o 
run_ci_tests_LDFLAGS=-lboost_prg_exec_monitor -lboost_system -lboost_unit_test_framework $(SUNDIALS_NVECTOR_SO) -L$(INSTALL_PATH)/lib -ldolfin

#######################################
# Other variables
#
# The list of output binary files to link
BUILD_TARGETS = $(addprefix $(MODULE_DIR)/,$(MODULES))
# The list of source files to be compiled
ifneq ($(findstring /.pixi/,$(PYTHON_PREFIX)),)
SOURCES = $(shell find src/llg src/util src/sundials -name \*.cc)
else
SOURCES = $(shell find src tests -name \*.cc)
endif
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
EIGEN_INCLUDE_DIR ?= $(shell if [ -d "$(PYTHON_PREFIX)/include/eigen3" ]; then echo "$(PYTHON_PREFIX)/include/eigen3"; else echo "/usr/include/eigen3"; fi)
CFLAGS_OPT ?= -O3 -funroll-loops -mfpmath=sse -march=native
ifeq ($(PYTHON_CONFIG_EXISTS),1)
CFLAGS_PYTHON=$(shell $(PYTHON_CONFIG) --includes) -I$(EIGEN_INCLUDE_DIR) -fPIC $(shell $(PYTHON_CONFIG) --ldflags) -l$(BOOST_PYTHON_LIB)
else
# Modern managed Python environments may omit python-config even though the
# headers and libpython are present through sysconfig. [Codex GPT-5.4]
PYTHON_INCLUDES=$(shell $(PYTHON) -c 'import sysconfig; print("-I{}".format(sysconfig.get_config_var("INCLUDEPY")))')
PYTHON_LDFLAGS=$(shell $(PYTHON) -c 'import sysconfig; libdir = sysconfig.get_config_var("LIBDIR"); ldversion = sysconfig.get_config_var("LDVERSION") or sysconfig.get_config_var("VERSION"); print("-L{} -lpython{}".format(libdir, ldversion))')
CFLAGS_PYTHON=$(PYTHON_INCLUDES) -I$(EIGEN_INCLUDE_DIR) -fPIC $(PYTHON_LDFLAGS) -l$(BOOST_PYTHON_LIB)
endif
# -rdynamic keeps private symbols in .so files, providing more detailed stack traces
# -MMD -MP generates .d dependency files
CFLAGS_OTHER=-g -rdynamic -fopenmp -MMD -MP -std=c++0x
# The historical system builds used -Werror successfully, but modern managed
# environments surface many third-party header warnings outside Finmag's own
# code. Keep strict warnings on the legacy path and relax them for pixi-style
# prefixes. [Codex GPT-5.4]
ifneq ($(findstring /.pixi/,$(PYTHON_PREFIX)),)
CFLAGS_WARN=-Wall
else
CFLAGS_WARN=-Wall -Werror -Wno-error=unused-function -Wno-error=deprecated-declarations
endif
# Add the source code directory to the include path
# Managed environments such as pixi place third-party headers under the
# environment prefix instead of /usr/include, so add that include root too. [Codex GPT-5.4]
CFLAGS_INCLUDE=-I$(abspath src) -I$(PYTHON_PREFIX)/include -I$(NUMPY_INCLUDE_DIR)
CFLAGS_NOPCH=$(CFLAGS_OPT) $(CFLAGS_PYTHON) $(CFLAGS_OTHER) $(CFLAGS_WARN) $(CFLAGS_INCLUDE)
CFLAGS=$(CFLAGS_NOPCH) $(CFLAGS_USE_PCH)

# The number of make jobs to run in parallel
N_SIMULTANEOUS_JOBS:=$(shell grep -ci processor /proc/cpuinfo)

FINMAGDIST_REVISION ?= HEAD

#REVISION_NR = $(shell hg id -n -r ${FINMAGDIST_REVISION})
REVISION_ID = $(shell git rev-parse ${FINMAGDIST_REVISION})

# TODO: This is probably not the best place for this target, but at
#       least it ensures that the version is re-generated whenever the
#       native code is compiled, so we are always up to date.
add_version: .add_version
.add_version: $(BUILD_TARGETS)
	@echo "FINMAG_VERSION: ${REVISION_ID}"
	@echo '__version__ = "${REVISION_ID}"' > ../src/finmag/__version__.py
	touch .add_version

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
build-targets : $(BUILD_TARGETS) add_version

# Compiles the source files into object files
object-files : $(OBJECT_FILES)

# Builds subprojects that have separate makefiles
subprojects :

# Build the custom nvector_serial library
nvec_serial :
	@$(MAKE) -C src/sundials/nvec_serial

# Builds an object file from a .cc file
%.o : %.cc $(PRECOMPILED_HEADER_FILE) Makefile
	@echo Building object file: $@
	@$(CXX) -c $< $(CFLAGS) $(MODULE_FLAGS) -o $@ ${UFC_INC}

# Creates the precompiled header file
pch: $(PRECOMPILED_HEADER_FILE)
$(PRECOMPILED_HEADER_FILE) : src/finmag_includes.h Makefile
	@echo Building precompiled header file: $@
	@mkdir -p $(PRECOMPILED_HEADER_DIR)
	@$(CXX) $(CFLAGS_NOPCH) -o $@ -c $< ${UFC_INC}

# Runs Boost.Test unit tests in native code
run-ci-tests:
	$(MAKE) -j $(NUM_PROCS) $(UNIT_TEST_BINARY)
	mkdir -p $(UNIT_TEST_RESULTS_DIR)
	./$(UNIT_TEST_BINARY) --output_format=XML --report_level=no --log_level=test_suite > $(UNIT_TEST_RESULTS_FILE)

# Removes all temporary files
clean:
	rm -f $(BUILD_TARGETS) $(PRECOMPILED_HEADER_FILE)
	find . \( -name \*.o -o -name \*.d -o -name \*.o.nm -o -name \*.gen.cc -o -name \*.gen.h -o -name \*.so \) -exec rm {} +
	rm -f ./src/sundials/nvec_serial/.nvector_serial_custom_malloc_compiled
	rm -f .add_version

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
