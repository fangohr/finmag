#
# Makefile to help putting a compiled version of finmag together
#

# (Temporary) directory where the binary distribution will be built
# and the tarball be placed. This can be configured by the user
FINMAGDIST_TMPDIR ?= /tmp/

FINMAGDIST_BUILD_DIR = ${FINMAGDIST_TMPDIR}/finmag
FINMAGDIST_DEST_TARBALL = ${FINMAGDIST_TMPDIR}/FinMag-dist.tar.bz2

info:
	echo "Makefile to help putting a compiled version of finmag together"

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
	cp pytest.ini ${FINMAGDIST_BUILD_DIR}

test:
	cd ${FINMAGDIST_BUILD_DIR} && PYTHONPATH=${FINMAGDIST_BUILD_DIR} py.test --verbose -x

bundle:
	cd ${FINMAGDIST_BUILD_DIR} && tar cvjf ${FINMAGDIST_DEST_TARBALL} \
		--exclude='*test*' \
		--exclude=1d.nmesh \
		--exclude=run_nmag_Eexch_log.log* \
		--exclude=Makefile \
		--exclude=distcp.* \
		--exclude=run_nmag_Eexch.py \
		--exclude=setup2.so \
		--exclude=__init__.pyc \
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
