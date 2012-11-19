# Paths that have to be added to PYTHONPATH prior to running the tests from here (for jenkins)
PYTHON_ROOTS = $(abspath ../../../src)

all: results.rst

results.rst:
	@echo "Running timings"
	PYTHONPATH=$(PYTHON_ROOTS) python run_finmag.py
