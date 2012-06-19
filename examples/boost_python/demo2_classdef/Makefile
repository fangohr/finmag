CFLAGS_PYTHON=-I/usr/include/python2.7 -fPIC -lboost_python -lpython2.7
CFLAGS=$(CFLAGS_PYTHON)

demo2_module.so : demo2_module.cc
	$(CXX) $^ $(CFLAGS) -shared -o $@
