/**
 * FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
 * Copyright (C) 2012 University of Southampton
 * Do not distribute
 *
 * CONTACT: h.fangohr@soton.ac.uk
 *
 * AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)
 */
#ifndef __FINMAG_INCLUDES_H
#define __FINMAG_INCLUDES_H

// Standard C/C++ includes
#include <cstring>
#include <cmath>
#include <cstdlib>

// The array-only native surface (e.g. the Task 11a FK BEM `bem_arrays` module)
// is compiled with -DFINMAG_NO_DOLFIN / -DFINMAG_NO_SUNDIALS so it can build in
// the DOLFINx environment where neither legacy DOLFIN nor SUNDIALS is present.
// In the legacy build these standard headers arrive transitively through
// dolfin.h; pull them in explicitly when dolfin.h is skipped. [Claude Opus 4.8]
#ifdef FINMAG_NO_DOLFIN
#include <iostream>
#include <ostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <utility>
#include <memory>
#include <csignal>
#endif

// OpenMP
#include <omp.h>

// Boost includes
#ifndef IDE_ERROR_BLOCK
#include <boost/mpl/map.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/list.hpp>
#include <boost/python.hpp>
#include <boost/typeof/typeof.hpp>
#include <boost/thread.hpp>
#endif

// CVODE/Sundials
#ifndef FINMAG_NO_SUNDIALS
#include <cvode/cvode.h>
#endif
// Dolfin
#ifndef FINMAG_NO_DOLFIN
#include <dolfin.h>
#endif

#endif
