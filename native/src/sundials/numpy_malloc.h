/**
 * FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
 * Copyright (C) 2012 University of Southampton
 * Do not distribute
 *
 * CONTACT: h.fangohr@soton.ac.uk
 *
 * AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)
 */

#ifndef __FINMAG_UTIL_SUNDIALS_NUMPY_MALLOC_H
#define __FINMAG_UTIL_SUNDIALS_NUMPY_MALLOC_H

#include "util/np_array.h"
#include <nvector/nvector_serial.h>

namespace finmag { namespace sundials {
    void register_numpy_malloc();

    np_array<double> nvector_to_array(N_Vector p);

    /* Wrapper class for Sundials NVectorSerial */
    class array_nvector {
    public:
        array_nvector(const np_array<double> &data);

        N_Vector ptr() { return vec; }

        ~array_nvector() {
            if (vec) {
                N_VDestroy(vec);
                vec = 0;
            }
        }
    private:
        // Disallow copy constructor & assignment
        // Use auto_ptr/unique_ptr/shared_ptr for shared nvector_serial objects
        array_nvector(const array_nvector&);
        void operator=(const array_nvector&);

        N_Vector vec;
        // store a reference to the original array to prevent array memory from being freed
        np_array<double> arr;
    };
}}

#endif
