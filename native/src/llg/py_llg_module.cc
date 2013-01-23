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

#include "llg.h"
#include "bem.h"
#include "heun.h"

#include "util/swig_dolfin.h"

BOOST_PYTHON_MODULE(llg)
{
    initialise_np_array();
    finmag::util::register_dolfin_swig_converters();

    bp::scope().attr("__doc__") = "C++ routines for computing the LLG (dM/dt and the Jacobean)";

    finmag::llg::register_llg();
    finmag::llg::register_bem();
    finmag::llg::register_heun();
}
