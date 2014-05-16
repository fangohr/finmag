CFLAGS_PYTHON=-I/usr/include/python2.7 -fPIC -lboost_python -lpython2.7 -ldolfin
CFLAGS=$(CFLAGS_PYTHON) -I/usr/include/eigen3

demo5_module.so : demo5_module.cc ../../../native/src/util/swig_dolfin.cc
	$(CXX) $^ $(CFLAGS) -shared -o $@
