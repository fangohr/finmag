info:
	echo "Make file to help putting a compiled version of finmag together"


libs:
	python setup-cython.py build_ext --inplace

#copy to alternative location
cp:
	python distcp.py
	cp -av finmag/tests /tmp/finmag/finmag/tests
	cd /tmp/finmag/finmag/tests && find . -name "*.so" | xargs rm -f
	cd /tmp/finmag/finmag/tests && find . -name "*.pyc" | xargs rm -f
	cd /tmp/finmag/finmag/tests && find finmag -name __pychache__ | xargs rm -fr	
	
test:
	cd /tmp/finmag/finmag/tests && PYTHONPATH=/tmp/finmag py.test -vx

clean:
	find finmag -name *.so | xargs rm -f		
	find finmag -name __pychache__ | xargs rm -fr	