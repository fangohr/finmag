/**
 * FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
 * Copyright (C) 2012 University of Southampton
 * Do not distribute
 *
 * CONTACT: h.fangohr@soton.ac.uk
 *
 * AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)
 */

#ifndef __FINMAG_UTIL_SUNDIALS_NVECTOR_CUSTOM_MALLOC_IMPL_H
#define __FINMAG_UTIL_SUNDIALS_NVECTOR_CUSTOM_MALLOC_IMPL_H

#include <stdlib.h>

#include "../nvector_custom_malloc.h"


#ifdef __cplusplus
extern "C" {
#endif

static void * (*nvector_custom_data_malloc)(size_t len, size_t el_size);
static void (*nvector_custom_data_free)(void *ptr);

void set_nvector_custom_data_malloc(void * (*malloc_func)(size_t, size_t),  void (*free_func)(void *)) {
    nvector_custom_data_malloc = malloc_func;
    nvector_custom_data_free = free_func;
}

#ifdef __cplusplus
}
#endif

#endif


