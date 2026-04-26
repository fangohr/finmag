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
#include <sundials/sundials_config.h>
#include <nvector/nvector_serial.h>
#if SUNDIALS_VERSION_MAJOR >= 7
#include <sundials/sundials_context.h>
#endif

namespace finmag { namespace sundials {
    namespace detail {
#if SUNDIALS_VERSION_MAJOR >= 7
        inline SUNContext default_suncontext() {
            static SUNContext sunctx = []() {
                SUNContext ctx = NULL;
                if (SUNContext_Create(SUN_COMM_NULL, &ctx) != SUN_SUCCESS) {
                    throw std::runtime_error("SUNContext_Create failed");
                }
                return ctx;
            }();
            return sunctx;
        }

        inline N_Vector new_serial_nvector(sunindextype len) {
            return N_VNew_Serial(len, default_suncontext());
        }

        inline N_Vector make_serial_nvector(sunindextype len, sunrealtype *data) {
            return N_VMake_Serial(len, data, default_suncontext());
        }
#else
        inline N_Vector new_serial_nvector(long int len) {
            return N_VNew_Serial(len);
        }

        inline N_Vector make_serial_nvector(long int len, realtype *data) {
            return N_VMake_Serial(len, data);
        }
#endif
    }

    void register_numpy_malloc();

    np_array<double> nvector_to_array(N_Vector p);

    bp::object nvector_to_array_object(N_Vector p);

    extern "C" void * numpy_malloc(size_t len, size_t el_size);
    extern "C" void numpy_free(void *ptr);

    /* Wrapper class for Sundials NVectorSerial */
    class array_nvector {
    public:
        array_nvector(const np_array<double> &data);

        N_Vector ptr() { return vec; }

        ~array_nvector() {
            if (vec) {
                N_VDestroy_Serial(vec);
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
