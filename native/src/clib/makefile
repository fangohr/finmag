all:
	python setup.py build_ext --inplace
	install clib.so ../../../src/finmag/native

clean:
	rm -rf clib.so build clib.c
	rm ../../../src/finmag/native/clib.so
