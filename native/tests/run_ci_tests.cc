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

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE finmag_native
#include "boost/test/unit_test.hpp"
#include "boost/test/unit_test_monitor.hpp"

#include "util/np_array.h"

// Convert Python exceptions into Boost.Test errors
void translate_python_exception(bp::error_already_set e) {
    PyObject *ptype = 0, *pvalue = 0, *ptraceback = 0;

    PyErr_Fetch(&ptype, &pvalue, &ptraceback);

    bp::object type, value, traceback;
    if (ptype) type = bp::object(bp::handle<>(ptype));
    if (pvalue) value = bp::object(bp::handle<>(pvalue));
    if (ptraceback) traceback = bp::object(bp::handle<>(ptraceback));

    std::string message;
    if (!value.is_none()) {
        message = bp::extract<std::string>(value);
    } else {
        message = "Unknown Python exception (value was None)";
    }

    BOOST_ERROR(message.c_str());
}

struct py_initialize {
    py_initialize() { Py_Initialize(); }
    ~py_initialize() { Py_Finalize(); }
};

struct init_python
{
    init_python() {
        main_module = bp::import("__main__");
        main_namespace = main_module.attr("__dict__");
        numpy = bp::import("numpy");
        boost::unit_test::unit_test_monitor.register_exception_translator<bp::error_already_set>(&translate_python_exception);
        initialise_np_array();
        // Note: the main_module/main_namespace/numpy objects must be deallocated before calling Py_Finalize.
        // This now happens automatically since the the py_init definition is above the bp::object definition
        // and therefore py_init is the last member variable to be destroyed
    }

    py_initialize py_init;

    bp::object main_module, main_namespace, numpy;
};

BOOST_GLOBAL_FIXTURE( init_python );
