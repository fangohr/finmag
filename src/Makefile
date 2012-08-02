info:
	echo "Make file to help putting a compiled version of finmag together"

libs:
	python setup-cython.py build_ext --inplace
	python setup-native.py # builds native/ .so files. 

#copy to alternative location
cp:
	python distcp.py /tmp/finmag/
	cp pytest.ini /tmp/finmag/

test:
	cd /tmp/finmag/ && PYTHONPATH=/tmp/finmag py.test --verbose

bundle:
	cd /tmp/finmag && tar cvjf /tmp/FinMag-dist.tar.bz2 \
		--exclude='*test*' \
		--exclude=1d.nmesh \
		--exclude=run_nmag_Eexch_log.log* \
		--exclude=Makefile \
		--exclude=distcp.* \
		--exclude=*.pyc \
		--exclude=setup2.so \
		--exclude=__pycache__ .

cleansrc:
	find finmag -name *.so | xargs rm -f		
	find finmag -name *.pyc | xargs rm -f		
	find finmag -name __pychache__ | xargs rm -fr	

cleanbin:
	cd /tmp/finmag/finmag && find . -name "*.so" | xargs rm -f
	cd /tmp/finmag/finmag && find . -name "*.pyc" | xargs rm -f
	cd /tmp/finmag/finmag && find . -name __pychache__ | xargs rm -fr	

removebin:
	rm -rf /tmp/finmag
	rm -f /tmp/FinMag-dist.tar.bz2

cleanall:
	make removebin
	make cleansrc

