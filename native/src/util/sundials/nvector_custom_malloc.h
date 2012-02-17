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

extern void set_nvector_custom_data_malloc(void * (*malloc_func)(size_t len, size_t el_size),  void (*free_func)(void *));

#ifdef __cplusplus
}
#endif

#endif


