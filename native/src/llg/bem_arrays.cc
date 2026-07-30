/**
 * FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
 * Copyright (C) 2012 University of Southampton
 * Do not distribute
 *
 * CONTACT: h.fangohr@soton.ac.uk
 *
 * Standalone, array-only Fredkin-Koehler / GCR BEM native module (Task 11a).
 *
 * This module exposes the array-based BEM/Lindholm surface used by FK demag
 * with NO dependency on libdolfin, the SWIG-DOLFIN mesh converters, or dolfin
 * headers. It is compiled with -DFINMAG_NO_DOLFIN / -DFINMAG_NO_SUNDIALS so it
 * builds and imports on Python 3.12 in the DOLFINx environment without legacy
 * DOLFIN or SUNDIALS installed. The DOLFIN BoundaryMesh entry points remain in
 * the legacy `llg` module. [Claude Opus 4.8]
 */

#include "finmag_includes.h"

#include "util/np_array.h"

#include "bem_arrays.h"

namespace finmag { namespace llg {

    // Registers only the array-based BEM surface (no dolfin coupling).
    void register_bem_arrays() {
        bp::def("compute_bem_fk_from_arrays", &compute_bem_from_arrays<true>);
        bp::def("compute_bem_gcr_from_arrays", &compute_bem_from_arrays<false>);
        bp::def("compute_lindholm_L", &compute_lindholm_formula<true>);
        bp::def("compute_lindholm_K", &compute_lindholm_formula<false>);
    }
}}

BOOST_PYTHON_MODULE(bem_arrays)
{
    initialise_np_array();

    bp::scope().attr("__doc__") =
        "Array-only FK/GCR BEM routines (no libdolfin dependency)";

    finmag::llg::register_bem_arrays();
}
