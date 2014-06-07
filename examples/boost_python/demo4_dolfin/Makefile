CFLAGS_PYTHON=-I/usr/include/python2.7 -fPIC -lboost_python -lpython2.7 -ldolfin
CFLAGS=$(CFLAGS_PYTHON) -I/usr/include/eigen3 -std=c++0x

demo4_module.so : demo4_module.cc ../../../native/src/util/swig_dolfin.cc
	$(CXX) $^ $(CFLAGS) -shared -o $@
