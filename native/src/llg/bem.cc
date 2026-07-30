/**
 * FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
 * Copyright (C) 2012 University of Southampton
 * Do not distribute
 *
 * CONTACT: h.fangohr@soton.ac.uk
 *
 * AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)
 */

#include "finmag_includes.h"

#include "util/np_array.h"

#include "vector3.h"

// The array-only Lindholm / BEM kernels now live in a shared, dolfin-free
// header so the standalone `bem_arrays` module can reuse them without linking
// libdolfin. This translation unit keeps the legacy DOLFIN BoundaryMesh entry
// points and the combined `register_bem` registration. [Claude Opus 4.8]
#include "bem_arrays.h"

namespace finmag { namespace llg {
    namespace df = dolfin;
    namespace vector = finmag::vector;

    // Returns a tuple (BEM matrix, boundary-mesh-to-global-mesh vertex index mapping)
    // If ComputeDoubleLayerPotential == true, computes the FK BEM
    // If ComputeDoubleLayerPotential == false, computes the GCR BEM
    template<bool ComputeDoubleLayerPotential>
    bp::object compute_bem(const std::shared_ptr<df::BoundaryMesh> bm_ptr) {
        df::BoundaryMesh &bm = *bm_ptr;
        ASSERT(bm.geometry().dim() == 3);
        ASSERT(bm.topology().dim() == 2);

        int n = bm.num_vertices();
        np_array<double> bem(n, n);
        // compute the boundary-to-global index mapping
        np_array<int> b2g_map(n);
        auto values = bm.entity_map(0).values();
        for (int i = 0; i < n; i++) b2g_map.data()[i] = values[i];

        // compute the BEM
        auto &geom = bm.geometry();

        // Loop through vertices of the mesh
        int n_vertices = bm.num_vertices();
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < n_vertices; i++) {
            vector::vector3 R(geom.point(i));

            double *bem_row = bem(i);

            // loop over all triangles on the surface mesh
            for (df::CellIterator c(bm); !c.end(); ++c) {
                // The cell must be a triangle
                if (c->num_entities(0) != 3) throw std::runtime_error("BEM computation: all cells in the boundary mesh must be triangles");

                // Get the 3 vertices
                int j_1 = c->entities(0)[0];
                int j_2 = c->entities(0)[1];
                int j_3 = c->entities(0)[2];
                vector::vector3 R1(geom.point(j_1));
                vector::vector3 R2(geom.point(j_2));
                vector::vector3 R3(geom.point(j_3));

                // Add the contribution of this triangle to B[i, j]
                std::pair<vector::vector3, double> L = lindholm_formula<ComputeDoubleLayerPotential>(R, R1, R2, R3);
                // We have to change sign for the single layer potential
                // since \int 1/|R-r| has a negative normal derivative across the boundary
                double factor = ComputeDoubleLayerPotential ? 1 : -1;
                bem_row[j_1] += factor*L.first[0];
                bem_row[j_2] += factor*L.first[1];
                bem_row[j_3] += factor*L.first[2];

                if (ComputeDoubleLayerPotential) {
                    // Add the solid angle term
                    bem_row[i] += L.second*(1./(4.*M_PI));
                }
            }
        }

        if (ComputeDoubleLayerPotential) {
            // Subtract 1 from the diagonal
            for (int i = 0; i < n; i++) bem(i)[i] -= 1.;
        }

        return bp::make_tuple(bem, b2g_map);
    }

    // The array-based `compute_bem_from_arrays` and `compute_lindholm_formula`
    // kernels are defined in the shared, dolfin-free `bem_arrays.h` header and
    // reused verbatim here so the legacy `register_bem` surface is unchanged.
    // [Claude Opus 4.8]

    void register_bem() {
        bp::def("compute_bem_fk", &compute_bem<true>);
        bp::def("compute_bem_gcr", &compute_bem<false>);
        bp::def("compute_bem_fk_from_arrays", &compute_bem_from_arrays<true>);
        bp::def("compute_bem_gcr_from_arrays", &compute_bem_from_arrays<false>);
        bp::def("compute_lindholm_L", &compute_lindholm_formula<true>);
        bp::def("compute_lindholm_K", &compute_lindholm_formula<false>);
    }
}}
