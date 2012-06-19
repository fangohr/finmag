CFLAGS_PYTHON=-I/usr/include/python2.7 -fPIC -lboost_python -lpython2.7
CFLAGS=$(CFLAGS_PYTHON)

demo1_module.so : demo1_module.cc
	$(CXX) $^ $(CFLAGS) -shared -o $@
