/**
 * FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
 * Copyright (C) 2012 University of Southampton
 * Do not distribute
 *
 * CONTACT: h.fangohr@soton.ac.uk
 *
 * AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)
 */

#include "../finmag_includes.h"

#include "swig_python_impl.h"

namespace finmag {
    namespace util {
        void register_dolfin_swig_converters() {
            // Register the Mesh hierarchy
            // Unfortunately, the Mesh hierarchy does not declare a virtual destructor (!)
            // TODO: not use MPL here
            typedef boost::mpl::list<
                dolfin::BoxMesh,
                dolfin::IntervalMesh,
                dolfin::RectangleMesh,
                dolfin::SubMesh,
                dolfin::UnitCircleMesh,
                dolfin::UnitCubeMesh,
                dolfin::UnitIntervalMesh,
                dolfin::UnitSquareMesh,
                dolfin::UnitTetrahedronMesh,
                dolfin::UnitTriangleMesh,
                dolfin::BoundaryMesh
            >::type derived_classes;

            register_swig_boost_shared_ptr_hierarchy<dolfin::Mesh, derived_classes>();

            register_swig_boost_shared_ptr<dolfin::BoundaryMesh>();
	  
	    register_swig_boost_shared_ptr<dolfin::GenericVector>();
        }
}}
