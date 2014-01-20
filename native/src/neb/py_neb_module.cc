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

#include "util/np_array.h"

#include "helper.h"

BOOST_PYTHON_MODULE(neb)
{
    initialise_np_array();

    bp::scope().attr("__doc__") = "C++ helper functions for NEB";

    finmag::neb::register_neb();
}
