# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

#######################################
# This makefile compiles the custom nvector_serial library
#
# Adjustable options are below.
#
CFLAGS=-O3 -fPIC -mfpmath=sse -march=native -ffast-math -lblas -llapack -lm -g -rdynamic
DEPS = Makefile nvector_custom_malloc_impl.h ../nvector_custom_malloc.h
SOURCES = nvector_serial_custom_malloc.c sundials_math.c

nvector_serial_custom_malloc.so : $(SOURCES) $(DEPS)
	gcc $(SOURCES) $(CFLAGS) -shared -o ../../../../src/finmag/native/lib$@

# This target can be used to print Makefile variables (such as PROJECT_DIR)
# from the command line, for example by saying 'make print-PROJECT_DIR'.
print-%:
	@echo $($*)
