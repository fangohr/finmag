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
# The file name of the output extension module
FINMAG_MODULE = ../src/finmag/native/finmag.so
# The list of source files to be compiled
SOURCES = src/py_finmag_module.cc src/util/np_array.cc
# The directory to store the precompiled header file.
PRECOMPILED_HEADER_DIR ?= /tmp/$(USER)/finmag-pch
# The list of output binary files to link
BUILD_TARGETS = $(FINMAG_MODULE)

#######################################
# Other variables
#
# The name of the precompiled header file.
PRECOMPILED_HEADER_FILE=$(PRECOMPILED_HEADER_DIR)/finmag_includes.h.gch
CFLAGS_USE_PCH = -include $(PRECOMPILED_HEADER_DIR)/finmag_includes.h
# Compilation flags
CFLAGS_OPT=-O3 -funroll-loops -mfpmath=sse -march=native
CFLAGS_PYTHON=-I/usr/include/python2.7 -fPIC -lboost_python -lpython2.7
# -rdynamic keeps private symbols in .so files, providing more detailed stack traces
# -MMD -MP generates .d dependency files
CFLAGS_OTHER=-g -rdynamic -fopenmp -MMD -MP
# Treat all warnings as errors
CFLAGS_WARN=-Wall -Werror
# Add the source code directory to the include path
CFLAGS_INCLUDE=-I$(abspath src)
CFLAGS_NOPCH=$(CFLAGS_OPT) $(CFLAGS_PYTHON) $(CFLAGS_OTHER) $(CFLAGS_WARN) $(CFLAGS_INCLUDE)
CFLAGS=$(CFLAGS_NOPCH) $(CFLAGS_USE_PCH)

# The list of object files to be compiled
OBJECT_FILES = $(patsubst %.cc,%.o,$(SOURCES))
# The number of make jobs to run in parallel
N_SIMULTANEOUS_JOBS:=$(shell grep -ci processor /proc/cpuinfo)

# Builds the whole project
parallel:
	@echo ------------------
	@echo Building object files: $(OBJECT_FILES)
	@echo ------------------
	make -j $(N_SIMULTANEOUS_JOBS) object-files
	@echo ------------------
	@echo Linking: $(BUILD_TARGETS)
	@echo ------------------
	make -j $(N_SIMULTANEOUS_JOBS) build-targets

# Links the object files and creates the binaries
build-targets : $(BUILD_TARGETS)

# Compiles the source files into object files
object-files : $(OBJECT_FILES)

# Builds an object file from a .cc file
%.o : %.cc $(PRECOMPILED_HEADER_FILE)
	@echo Building object file: $@
	@$(CXX) -c $< $(CFLAGS) $(MODULE_FLAGS) -o $@

# Creates the precompiled header file
pch: $(PRECOMPILED_HEADER_FILE)
$(PRECOMPILED_HEADER_FILE) : src/finmag_includes.h
	@echo Building precompiled header file: $@
	@mkdir -p $(PRECOMPILED_HEADER_DIR)
	@$(CXX) $(CFLAGS_NOPCH) -o $@ -c $<

# Removes all temporary files
clean:
	rm -f $(BUILD_TARGETS) $(PRECOMPILED_HEADER_FILE)
	find . \( -name \*.o -o -name \*.d -o -name \*.o.nm -o -name \*.gen.cc -o -name \*.gen.h \) -exec rm {} +

$(FINMAG_MODULE) : $(OBJECT_FILES)
	@echo Building shared library: $@
	$(CXX) $^ $(CFLAGS) -shared -o $@

.PHONY: clean pch all object-files build-targets parallel
