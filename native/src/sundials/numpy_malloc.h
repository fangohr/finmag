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

    /* Wrapper class for Sundials NVectorSerial */
    class nvector_serial {
    public:
        nvector_serial(const np_array<double> &data);

        static np_array<double> to_np_array(NVector *p);

        NVector ptr() { return vec; }
    private:
        // Disallow copy constructor & assignment
        // Use auto_ptr/unique_ptr/shared_ptr for shared nvector_serial objects
        nvector_serial(const nvector_serial&);
        void operator=(const nvector_serial&);

        NVector vec;
    };
}}

#endif
