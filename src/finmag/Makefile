# (Temporary) directory where the binary distribution will be built
# and the tarball be placed.
FINMAGDIST_TMPDIR ?= /tmp

libs:
	python setup2.py build_ext --inplace

#copy to alternative location
cp:
	python distcp.py ${FINMAGDIST_TMPDIR}