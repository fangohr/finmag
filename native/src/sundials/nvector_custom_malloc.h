/**
 * FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
 * Copyright (C) 2012 University of Southampton
 * Do not distribute
 *
 * CONTACT: h.fangohr@soton.ac.uk
 *
 * AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)
 */

#ifndef __FINMAG_UTIL_SUNDIALS_NVECTOR_CUSTOM_MALLOC_H
#define __FINMAG_UTIL_SUNDIALS_NVECTOR_CUSTOM_MALLOC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <nvector/nvector_serial.h>

extern void set_nvector_custom_allocators(
            void * (*data_malloc_func)(size_t, size_t),
            void (*data_free_func)(void *),
            N_VectorContent_Serial (*nvec_malloc_func)(),
            void (*nvec_free_func)(N_VectorContent_Serial)
        );

#ifdef __cplusplus
}
#endif

#endif


