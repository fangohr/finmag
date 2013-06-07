#
# Makefile to help putting a compiled version of finmag together
#

# (Temporary) directory where the binary distribution will be built.
FINMAGDIST_TMPDIR ?= /tmp

# Directory where the final binary tarball will be placed.
FINMAGDIST_DESTDIR ?= $(FINMAGDIST_TMPDIR)

FINMAGDIST_BUILD_DIR = ${FINMAGDIST_TMPDIR}/finmag
FINMAGDIST_REVISION ?= tip
FINMAGDIST_TARBALL_SUFFIX ?=

# The tarball name is determined from the
#FINMAGDIST_DEST_TARBALL = $(shell python print_tarball_name.py ${FINMAGDIST_REVISION} "${FINMAGDIST_TARBALL_SUFFIX}" ${FINMAGDIST_DESTDIR} 2>/dev/null | tail -n1)
FINMAGDIST_DEST_TARBALL ?=

info:
	@echo "This is the Makefile to help putting a compiled version of finmag together"
	@echo "Output tarball name: ${FINMAGDIST_DEST_TARBALL}"

libs:
	python setup-cython.py build_ext --inplace
	python setup-native.py # builds native/ .so files. 
	python -m compileall *

cleansrc:
	find finmag -name "*.so" | xargs rm -f
	find finmag -name "*.pyc" | xargs rm -f
	find finmag -name "__pychache__" | xargs rm -fr

#copy to alternative location
cp:
	python distcp.py ${FINMAGDIST_BUILD_DIR}
	cp ../pytest.ini ${FINMAGDIST_BUILD_DIR}
	cp *.egg-info ${FINMAGDIST_BUILD_DIR}

test:
	@echo "Running target 'test' using PYTHONPATH=${FINMAGDIST_BUILD_DIR}"
	cd ${FINMAGDIST_BUILD_DIR} && PYTHONPATH=${FINMAGDIST_BUILD_DIR} py.test --verbose -x

bundle: info
ifeq "${FINMAGDIST_DEST_TARBALL}" ""
	    @echo "Error: FINMAGDIST_DEST_TARBALL is empty"
	    exit 1
endif
	cd ${FINMAGDIST_BUILD_DIR} && tar cvjf ${FINMAGDIST_DEST_TARBALL} \
		--exclude='*test*' \
		--exclude=1d.nmesh \
		--exclude=run_nmag_Eexch_log.log* \
		--exclude=Makefile \
		--exclude=distcp.* \
		--exclude=run_nmag_Eexch.py \
		--exclude=setup2.so \
		--exclude=__init__.pyc \
		--exclude=*.xml.gz \
		--exclude=*.geo \
		--exclude=unnamed.ndt \
		--exclude=_mat.xml \
		--exclude=_bi.xml \
		--exclude=unnamed.log\
		--exclude=__pycache__ . 

cleanbin:
	cd ${FINMAGDIST_BUILD_DIR}/finmag && find . -name "*.so" | xargs rm -f
	cd ${FINMAGDIST_BUILD_DIR}/finmag && find . -name "*.pyc" | xargs rm -f
	cd ${FINMAGDIST_BUILD_DIR}/finmag && find . -name __pycache__ | xargs rm -fr

removebin:
	rm -rf ${FINMAGDIST_BUILD_DIR}
	rm -f ${FINMAGDIST_DEST_TARBALL}

cleanall:
	make removebin
	make cleansrc

# This target can be used to print Makefile variables (such as FINMAGDIST_DESTDIR)
# from the command line, for example by saying 'make print-FINMAGDIST_DESTDIR'.
print-%:
	@echo $($*)
