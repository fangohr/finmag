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

namespace finmag { namespace llg {
    namespace df = dolfin;
    namespace vector = finmag::vector;

    namespace {
        inline double sq(double x) { return x*x; }

        // Computes the Lindholm formula as well as the solid angle for the specified points
        // If ComputeDoubleLayerPotential == true, computes the double layer potential
        // If ComputeDoubleLayerPotential == false, computes the single layer potential
        template<bool ComputeDoubleLayerPotential>
        std::pair<vector::vector3, double>
        lindholm_formula(const vector::vector3 &R, const vector::vector3 &R1, const vector::vector3 &R2, const vector::vector3 &R3) {
            using namespace vector;

            vector3 r1(R1, R);
            vector3 r2(R2, R);
            vector3 r3(R3, R);

            // s_i is the length of the i'th side
            // xi_hat is the unit vector for the i'th side
            double s_1 = (r2 - r1).length(); vector3 xi_hat_1  = (r2 - r1).normalized();
            double s_2 = (r3 - r2).length(); vector3 xi_hat_2  = (r3 - r2).normalized();
            double s_3 = (r1 - r3).length(); vector3 xi_hat_3  = (r1 - r3).normalized();
            // A_T is the area of the triangle
            double A_T = triangle_area(r1, r2, r3);
            // zeta_hat is the vector normal to the triangle
            vector3 zeta_hat = cross(r2-r1, r3-r1).normalized();
            // zeta is the distance from R to the triangle plane
            double zeta = dot(zeta_hat, r1);

            // eta_i is the distance to the i'th side projected in the triangle plane
            vector3 eta_hat_1 = cross(zeta_hat, xi_hat_1);
            vector3 eta_hat_2 = cross(zeta_hat, xi_hat_2);
            vector3 eta_hat_3 = cross(zeta_hat, xi_hat_3);
            double eta_1 = dot(eta_hat_1, r1);
            double eta_2 = dot(eta_hat_2, r2);
            double eta_3 = dot(eta_hat_3, r1);

            // gamma_i_j is the cosine angle between the (i+1)'th and j'th side
            vector3 gamma_1(
                dot(xi_hat_2, xi_hat_1),
                dot(xi_hat_2, xi_hat_2),
                dot(xi_hat_2, xi_hat_3)
            );
            vector3 gamma_2(
                dot(xi_hat_3, xi_hat_1),
                dot(xi_hat_3, xi_hat_2),
                dot(xi_hat_3, xi_hat_3)
            );
            vector3 gamma_3(
                dot(xi_hat_1, xi_hat_1),
                dot(xi_hat_1, xi_hat_2),
                dot(xi_hat_1, xi_hat_3)
            );

            // P is an auxiliary variable
            double r1_len = r1.length(), r2_len = r2.length(), r3_len = r3.length();
            //printf("r1_len=%f, r2_len=%f, r3_len=%f, s1=%f, s2=%f, s3=%f\n", r1_len, r2_len, r3_len, s_1, s_2, s_3);
            vector3 P(
                log((r1_len + r2_len + s_1) / (r1_len + r2_len - s_1 + 1e-300)),
                log((r2_len + r3_len + s_2) / (r2_len + r3_len - s_2 + 1e-300)),
                log((r3_len + r1_len + s_3) / (r3_len + r1_len - s_3 + 1e-300))
            );

            // Sigma_T is the solid angle subtended by the triangle as seen from R
            double Sigma_T = solid_angle(r1, r2, r3);

            if (ComputeDoubleLayerPotential) {
                return std::make_pair(
                    vector3(
                        s_2/A_T/(8*M_PI) * (eta_2 * Sigma_T - zeta * dot(gamma_1, P)),
                        s_3/A_T/(8*M_PI) * (eta_3 * Sigma_T - zeta * dot(gamma_2, P)),
                        s_1/A_T/(8*M_PI) * (eta_1 * Sigma_T - zeta * dot(gamma_3, P))
                    ),
                    Sigma_T
                );
            } else {
                double eta_dot_p = eta_1*P.x() + eta_2*P.y() + eta_3*P.z();

                vector3 Q(
                    (r1_len + r2_len)/(2*s_1)*(1+sq(r1_len - r2_len)/sq(s_1)),
                    (r2_len + r3_len)/(2*s_2)*(1+sq(r2_len - r3_len)/sq(s_2)),
                    (r3_len + r1_len)/(2*s_3)*(1+sq(r3_len - r1_len)/sq(s_3))
                );
                vector3 K(
                    (sq(zeta) + sq(eta_1))*P[0] + sq(s_1)*Q[0],
                    (sq(zeta) + sq(eta_2))*P[1] + sq(s_2)*Q[1],
                    (sq(zeta) + sq(eta_3))*P[2] + sq(s_3)*Q[2]
                );

                return std::make_pair(
                    vector3(
                        s_2/A_T/(8*M_PI) * (eta_2*(zeta*Sigma_T + eta_dot_p) - 0.5*dot(gamma_1, K)),
                        s_3/A_T/(8*M_PI) * (eta_3*(zeta*Sigma_T + eta_dot_p) - 0.5*dot(gamma_2, K)),
                        s_1/A_T/(8*M_PI) * (eta_1*(zeta*Sigma_T + eta_dot_p) - 0.5*dot(gamma_3, K))
                    ),
                    Sigma_T
                );
            }
        }
    }

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

    // DOLFIN 2019 exposes BoundaryMesh through a Python binding that no longer
    // round-trips through the legacy shared_ptr converter, so provide an
    // equivalent array-based entry point for the Python demag path. [Codex GPT-5.4]
    template<bool ComputeDoubleLayerPotential>
    bp::object compute_bem_from_arrays(np_array<double> coords, np_array<long> cells, np_array<long> b2g_input) {
        coords.check_ndim(2, "compute_bem_from_arrays: coords");
        cells.check_ndim(2, "compute_bem_from_arrays: cells");
        b2g_input.check_ndim(1, "compute_bem_from_arrays: b2g_map");

        if (coords.dim()[1] != 3) {
            throw std::invalid_argument("compute_bem_from_arrays: coords must have shape (n, 3)");
        }
        if (cells.dim()[1] != 3) {
            throw std::invalid_argument("compute_bem_from_arrays: cells must have shape (m, 3)");
        }
        if (coords.dim()[0] != b2g_input.dim()[0]) {
            throw std::invalid_argument("compute_bem_from_arrays: coords and b2g_map lengths must agree");
        }

        int n = coords.dim()[0];
        int n_cells = cells.dim()[0];
        np_array<double> bem(n, n);
        np_array<int> b2g_map(n);
        for (int i = 0; i < n; i++) {
            b2g_map.data()[i] = static_cast<int>(b2g_input.data()[i]);
        }

        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < n; i++) {
            vector::vector3 R(coords(i, 0)[0], coords(i, 1)[0], coords(i, 2)[0]);
            double *bem_row = bem(i);

            for (int c = 0; c < n_cells; c++) {
                int j_1 = static_cast<int>(cells(c, 0)[0]);
                int j_2 = static_cast<int>(cells(c, 1)[0]);
                int j_3 = static_cast<int>(cells(c, 2)[0]);

                vector::vector3 R1(coords(j_1, 0)[0], coords(j_1, 1)[0], coords(j_1, 2)[0]);
                vector::vector3 R2(coords(j_2, 0)[0], coords(j_2, 1)[0], coords(j_2, 2)[0]);
                vector::vector3 R3(coords(j_3, 0)[0], coords(j_3, 1)[0], coords(j_3, 2)[0]);

                std::pair<vector::vector3, double> L = lindholm_formula<ComputeDoubleLayerPotential>(R, R1, R2, R3);
                double factor = ComputeDoubleLayerPotential ? 1 : -1;
                bem_row[j_1] += factor * L.first[0];
                bem_row[j_2] += factor * L.first[1];
                bem_row[j_3] += factor * L.first[2];

                if (ComputeDoubleLayerPotential) {
                    bem_row[i] += L.second * (1. / (4. * M_PI));
                }
            }
        }

        if (ComputeDoubleLayerPotential) {
            for (int i = 0; i < n; i++) bem(i)[i] -= 1.;
        }

        return bp::make_tuple(bem, b2g_map);
    }

    // This function is only used for testing; use compute_bem to compute the BEM itself instead
    // If ComputeDoubleLayerPotential == true, computes the double layer potential
    // If ComputeDoubleLayerPotential == false, computes the single layer potential
    template<bool ComputeDoubleLayerPotential>
    np_array<double> compute_lindholm_formula(np_array<double> r, np_array<double> r1, np_array<double> r2, np_array<double> r3) {
        r.check_shape(3, "compute_lindholm_formula: r");
        r1.check_shape(3, "compute_lindholm_formula: r1");
        r2.check_shape(3, "compute_lindholm_formula: r2");
        r3.check_shape(3, "compute_lindholm_formula: r3");

        using namespace vector;

        np_array<double> res(3);
        std::pair<vector::vector3, double> L = lindholm_formula<ComputeDoubleLayerPotential>(vector3(r.data()), vector3(r1.data()), vector3(r2.data()), vector3(r3.data()));
        res.data()[0] = L.first[0];
        res.data()[1] = L.first[1];
        res.data()[2] = L.first[2];
        return res;
    }

    void register_bem() {
        bp::def("compute_bem_fk", &compute_bem<true>);
        bp::def("compute_bem_gcr", &compute_bem<false>);
        bp::def("compute_bem_fk_from_arrays", &compute_bem_from_arrays<true>);
        bp::def("compute_bem_gcr_from_arrays", &compute_bem_from_arrays<false>);
        bp::def("compute_lindholm_L", &compute_lindholm_formula<true>);
        bp::def("compute_lindholm_K", &compute_lindholm_formula<false>);
    }
}}
