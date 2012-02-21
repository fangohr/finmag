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

namespace finmag { namespace sundials {
    void register_numpy_malloc();

    np_array<double> nvector_to_array(NVector p);

    /* Wrapper class for Sundials NVectorSerial */
    class array_nvector {
    public:
        array_nvector(const np_array<double> &data);

        NVector ptr() { return vec; }

        ~array_nvector() {
            if (vec) {
                N_VDestroy(vec);
                vec = 0;
            }
        }
    private:
        // Disallow copy constructor & assignment
        // Use auto_ptr/unique_ptr/shared_ptr for shared nvector_serial objects
        nvector_serial(const nvector_serial&);
        void operator=(const nvector_serial&);

        NVector vec;
        // store a reference to the original array to prevent array memory from being freed
        np_array<double> arr;
    };
}}

#endif
