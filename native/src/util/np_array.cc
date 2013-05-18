/**
 * FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
 * Copyright (C) 2012 University of Southampton
 * Do not distribute
 *
 * CONTACT: h.fangohr@soton.ac.uk
 *
 * AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)
 */

#include "finmag_includes.h"
#include <execinfo.h>

// finmag_PyArray_API will be defined in this compilation unit
#define PY_ARRAY_UNIQUE_SYMBOL finmag_PyArray_API
// include arrayobject.h before np_array.h
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "np_array.h"

// C-to-python converter
template<class T>
struct np_array_to_python_converter
{
    static PyObject* convert(const np_array<T>& a) { return bp::incref((PyObject*) a.array()); };

    static PyTypeObject const* get_pytype() { return &PyArray_Type; }
};

struct np_array_initialiser
{
    template<class K, class V> void operator()(mpl::pair<K, V>  x) {
        // Hack to avoid double registrations...
        if (!bp::converter::registry::lookup(bp::type_id<np_array<K> > ()).m_to_python) {
            bp::to_python_converter<np_array<K>, np_array_to_python_converter<K>, true>();
        }
        np_array<K>::register_converter();
    }
};

// handler for SIGSEGV that prints a backtrace before crashing
static void sigsegv_handler(int sig)
{
    void *array[30];
    size_t size = backtrace(array, 30);

    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, 2);
    exit(1);
}

void initialise_np_array()
{
    // install the segmentation fault handler that prints a stack trace
    signal(SIGSEGV, sigsegv_handler);
    // import the Python array object
    import_array();
    // register the from-python converters
    mpl::for_each<numpy_types>(np_array_initialiser());
}

void assertion_failed(const char *msg, const char* file, int line) {
    std::cerr << "Assertion failed: " << msg << " at " << file << ":" << line << std::endl;
    exit(-1);
}
