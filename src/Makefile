info:
	echo "Make file to help putting a compiled version of finmag together"


libs:
	python setup-cython.py build_ext --inplace

#copy to alternative location
cp:
	python distcp.py
	

clean:
	find finmag -name *.so | xargs rm -f		
	find finmag -name __pychache__ | xargs rm -fr	