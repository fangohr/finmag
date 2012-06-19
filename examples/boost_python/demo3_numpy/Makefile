CFLAGS_PYTHON=-I/usr/include/python2.7 -fPIC -lboost_python -lpython2.7
CFLAGS=$(CFLAGS_PYTHON) -I.

demo3_module.so : demo3_module.cc ../../../native/src/util/np_array.cc
	$(CXX) $^ $(CFLAGS) -shared -o $@
