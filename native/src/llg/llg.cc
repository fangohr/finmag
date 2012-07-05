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

#include "llg.h"

#include "util/python_threading.h"

namespace finmag { namespace llg {
    namespace {
        // Components of a cross product
        inline double cross0(double a0, double a1, double a2, double b0, double b1, double b2) { return a1*b2 - a2*b1; }
        inline double cross1(double a0, double a1, double a2, double b0, double b1, double b2) { return a2*b0 - a0*b2; }
        inline double cross2(double a0, double a1, double a2, double b0, double b1, double b2) { return a0*b1 - a1*b0; }

        // TODO: Look up values.
        const double e = 1; // Electric charge of the electron.
        const double h_bar = 1; // Plank constant.
        const double mu_0 = 1; // Permeability of free space.

        void calc_llg_dmdt(
                const np_array<double> &m,
                const np_array<double> &H,
                double t,
                const np_array<double> &dmdt,
                const np_array<long> &pins,
                double gamma,
                const np_array<double> &alpha,
                double char_time,
                bool do_precession) {
            m.check_ndim(2, "calc_llg_dmdt: m");
            int n = m.dim()[1];

            m.check_shape(3, n, "calc_llg_dmdt: m");
            H.check_shape(3, n, "calc_llg_dmdt: H");
            dmdt.check_shape(3, n, "calc_llg_dmdt: dmdt");

            alpha.check_ndim(1, "calc_llg_dmdt: alpha");
            alpha.check_shape(n, "calc_llg_dmdt: alpha");

            double *m0 = m(0), *m1 = m(1), *m2 = m(2);
            double *h0 = H(0), *h1 = H(1), *h2 = H(2);
            double *dm0 = dmdt(0), *dm1 = dmdt(1), *dm2 = dmdt(2);
            double precession_coeff, damping_coeff, gamma_LL;
            double relax_coeff = 0.1/char_time;

            finmag::util::scoped_gil_release release_gil;

            // calculate dmdt
            #pragma omp parallel for schedule(guided)
            for (int i=0; i < n; i++) {

                gamma_LL = gamma / (1 + (pow(*alpha[i], 2)));
                precession_coeff = - gamma_LL;
                damping_coeff = - gamma_LL * (*alpha[i]);

                // add damping: m x (m x H) == (m.H)m - (m.m)H, multiplied by -gamma alpha
                double mh = m0[i] * h0[i] + m1[i] * h1[i] + m2[i] * h2[i];
                double mm = m0[i] * m0[i] + m1[i] * m1[i] + m2[i] * m2[i];
                dm0[i] = damping_coeff*(m0[i] * mh - h0[i] * mm);
                dm1[i] = damping_coeff*(m1[i] * mh - h1[i] * mm);
                dm2[i] = damping_coeff*(m2[i] * mh - h2[i] * mm);

                // add relaxation of |m|
                double relax = relax_coeff*(1.-mm);
                dm0[i] += relax*m0[i];
                dm1[i] += relax*m1[i];
                dm2[i] += relax*m2[i];

                if (do_precession) {
                    // add precession: - gamma m x H
                    dm0[i] += precession_coeff*cross0(m0[i], m1[i], m2[i], h0[i], h1[i], h2[i]);
                    dm1[i] += precession_coeff*cross1(m0[i], m1[i], m2[i], h0[i], h1[i], h2[i]);
                    dm2[i] += precession_coeff*cross2(m0[i], m1[i], m2[i], h0[i], h1[i], h2[i]);
                }
            }

            pins.check_ndim(1, "calc_llg_dmdt: pins");
            const int nb_pins = pins.dim()[0];
            for (int i = 0; i < nb_pins; i++) {
                dm0[*pins[i]] = 0;
                dm1[*pins[i]] = 0;
                dm2[*pins[i]] = 0;
            }
        }
    
        void calc_llg_slonczewski_dmdt(
                const np_array<double> &m,
                const np_array<double> &H,
                double t,
                const np_array<double> &dmdt,
                const np_array<long> &pins,
                double gamma,
                const np_array<double> &alpha,
                double char_time,
                bool do_precession,
                double J,
                double P,
                double d,
                double Ms,
                const np_array<double> &p) {
            m.check_ndim(2, "calc_llg_dmdt: m");
            int n = m.dim()[1];

            m.check_shape(3, n, "calc_llg_dmdt: m");
            H.check_shape(3, n, "calc_llg_dmdt: H");
            dmdt.check_shape(3, n, "calc_llg_dmdt: dmdt");

            alpha.check_ndim(1, "calc_llg_dmdt: alpha");
            alpha.check_shape(n, "calc_llg_dmdt: alpha");

            double *m0 = m(0), *m1 = m(1), *m2 = m(2);
            double *h0 = H(0), *h1 = H(1), *h2 = H(2);
            double *dm0 = dmdt(0), *dm1 = dmdt(1), *dm2 = dmdt(2);
            double precession_coeff, damping_coeff, gamma_LL;
            double relax_coeff = 0.1/char_time;

            p.check_shape(3, n, "calc_llg_dmdt: p");
            double *p0 = p(0), *p1 = p(1), *p2 = p(2); 
            double stt_pre_coeff = J * h_bar / (mu_0 * Ms * e * d);
            double a_P = 4 * pow(sqrt(P) / (1 + P), 3); 

            finmag::util::scoped_gil_release release_gil;

            // calculate dmdt
            #pragma omp parallel for schedule(guided)
            for (int i=0; i < n; i++) {

                gamma_LL = gamma / (1 + (pow(*alpha[i], 2)));
                precession_coeff = - gamma_LL;
                damping_coeff = - gamma_LL * (*alpha[i]);

                // add damping: m x (m x H) == (m.H)m - (m.m)H, multiplied by -gamma alpha
                double mh = m0[i] * h0[i] + m1[i] * h1[i] + m2[i] * h2[i];
                double mm = m0[i] * m0[i] + m1[i] * m1[i] + m2[i] * m2[i];
                dm0[i] = damping_coeff*(m0[i] * mh - h0[i] * mm);
                dm1[i] = damping_coeff*(m1[i] * mh - h1[i] * mm);
                dm2[i] = damping_coeff*(m2[i] * mh - h2[i] * mm);

                // add relaxation of |m|
                double relax = relax_coeff*(1.-mm);
                dm0[i] += relax*m0[i];
                dm1[i] += relax*m1[i];
                dm2[i] += relax*m2[i];

                if (do_precession) {
                    // add precession: - gamma m x H
                    dm0[i] += precession_coeff*cross0(m0[i], m1[i], m2[i], h0[i], h1[i], h2[i]);
                    dm1[i] += precession_coeff*cross1(m0[i], m1[i], m2[i], h0[i], h1[i], h2[i]);
                    dm2[i] += precession_coeff*cross2(m0[i], m1[i], m2[i], h0[i], h1[i], h2[i]);
                }

                // scalar product m p
                double mp = m0[i] * p0[i] + m1[i] * p1[i] + m2[i] * p2[i];
                // cross product m x p
                double mp0 = cross0(m0[i], m1[i], m2[i], p0[i], p1[i], p2[i]);
                double mp1 = cross1(m0[i], m1[i], m2[i], p0[i], p1[i], p2[i]);
                double mp2 = cross2(m0[i], m1[i], m2[i], p0[i], p1[i], p2[i]);

                // add Slonczewski spin-torque term
                double stt_coeff = - gamma_LL * stt_pre_coeff * a_P / (3 + mp - 4 * a_P);
                dm0[i] += *alpha[i] * stt_coeff * mp0 - stt_coeff * cross0(m0[i], m1[i], m2[i], mp0, mp1, mp2);
                dm1[i] += *alpha[i] * stt_coeff * mp1 - stt_coeff * cross1(m0[i], m1[i], m2[i], mp0, mp1, mp2);
                dm2[i] += *alpha[i] * stt_coeff * mp2 - stt_coeff * cross2(m0[i], m1[i], m2[i], mp0, mp1, mp2);
            }

            pins.check_ndim(1, "calc_llg_dmdt: pins");
            const int nb_pins = pins.dim()[0];
            for (int i = 0; i < nb_pins; i++) {
                dm0[*pins[i]] = 0;
                dm1[*pins[i]] = 0;
                dm2[*pins[i]] = 0;
            }
        }
    
        void calc_llg_jtimes(
                const np_array<double> &m,
                const np_array<double> &H,
                const np_array<double> &mp,
                const np_array<double> &Hp,
                double t,
                const np_array<double> &jtimes,
                double gamma_LL,
                double alpha,
                double char_time,
                bool do_precession) {

	   //* m - magnetisation
	   //* H - effective field
	   //* mp - the magnetisation vector to compute the product with (read it as m-prime, m')
	   //* Hp - the derivative of the effective field in the direction of mp, i.e. d H_eff(m + a m')/da | a = 0
	   //* jtimes - the output Jacobean product, i.e. d rhs(m + a m')/da | a = 0

            m.check_ndim(2, "calc_llg_dmdt: m");
            int n = m.dim()[1];

            m.check_shape(3, n, "calc_llg_dmdt: m");
            mp.check_shape(3, n, "calc_llg_dmdt: mp");
            H.check_shape(3, n, "calc_llg_dmdt: H");
            Hp.check_shape(3, n, "calc_llg_dmdt: Hp");
            jtimes.check_shape(3, n, "calc_llg_dmdt: jtimes");

            double precession_coeff = -gamma_LL;
            double damping_coeff = -gamma_LL*alpha;
            double *m0 = m(0), *m1 = m(1), *m2 = m(2);
            double *mp0 = mp(0), *mp1 = mp(1), *mp2 = mp(2);
            double *h0 = H(0), *h1 = H(1), *h2 = H(2);
            double *hp0 = Hp(0), *hp1 = Hp(1), *hp2 = Hp(2);
            double *jtimes0 = jtimes(0), *jtimes1 = jtimes(1), *jtimes2 = jtimes(2);
            double relax_coeff = 0.1/char_time;

            finmag::util::scoped_gil_release release_gil;

            if (do_precession) {
                #pragma omp parallel for schedule(guided)
                for (int i = 0; i < n; i++) {
                    // add precession: mp x H + m x Hp
                    jtimes0[i] = precession_coeff*(cross0(mp0[i], mp1[i], mp2[i], h0[i], h1[i], h2[i])
                            + cross0(m0[i], m1[i], m2[i], hp0[i], hp1[i], hp2[i]));
                    jtimes1[i] = precession_coeff*(cross1(mp0[i], mp1[i], mp2[i], h0[i], h1[i], h2[i])
                            + cross1(m0[i], m1[i], m2[i], hp0[i], hp1[i], hp2[i]));
                    jtimes2[i] = precession_coeff*(cross2(mp0[i], mp1[i], mp2[i], h0[i], h1[i], h2[i])
                            + cross2(m0[i], m1[i], m2[i], hp0[i], hp1[i], hp2[i]));

                    // add damping: m x (m x H) == (m.H)m - (m.m)H, multiplied by -gamma alpha
                    // derivative is [(mp.H) + (m.Hp)]m + (m.H)mp - 2(m.mp)H - (m.m)Hp
                    double mp_h_m_hp = mp0[i]*h0[i] + mp1[i]*h1[i] + mp2[i]*h2[i]
                            + m0[i]*hp0[i] + m1[i]*hp1[i] + m2[i]*hp2[i];
                    double m_h = m0[i] * h0[i] + m1[i] * h1[i] + m2[i] * h2[i];
                    double m_mp = -2.*(m0[i]*mp0[i] + m1[i]*mp1[i] + m2[i]*mp2[i]);
                    double m_m = -(m0[i] * m0[i] + m1[i] * m1[i] + m2[i] * m2[i]);

                    jtimes0[i] += damping_coeff*(mp_h_m_hp*m0[i] + m_h*mp0[i] + m_mp*h0[i] + m_m*hp0[i]);
                    jtimes1[i] += damping_coeff*(mp_h_m_hp*m1[i] + m_h*mp1[i] + m_mp*h1[i] + m_m*hp1[i]);
                    jtimes2[i] += damping_coeff*(mp_h_m_hp*m2[i] + m_h*mp2[i] + m_mp*h2[i] + m_m*hp2[i]);

                    // add relaxation of |m|: (1 - (m.m))m
                    // derivative is -2(m.mp)m + (1-(m.m))mp
                    jtimes0[i] += relax_coeff*(m_mp*m0[i] + (1.+m_m)*mp0[i]);
                    jtimes1[i] += relax_coeff*(m_mp*m1[i] + (1.+m_m)*mp1[i]);
                    jtimes2[i] += relax_coeff*(m_mp*m2[i] + (1.+m_m)*mp2[i]);
                }
            } else {
                #pragma omp parallel for schedule(guided)
                for (int i = 0; i < n; i++) {
                    // add damping: m x (m x H) == (m.H)m - (m.m)H, multiplied by -gamma alpha
                    // derivative is [(mp.H) + (m.Hp)]m + (m.H)mp - 2(m.mp)H - (m.m)Hp
                    double mp_h_m_hp = mp0[i]*h0[i] + mp1[i]*h1[i] + mp2[i]*h2[i]
                            + m0[i]*hp0[i] + m1[i]*hp1[i] + m2[i]*hp2[i];
                    double m_h = m0[i] * h0[i] + m1[i] * h1[i] + m2[i] * h2[i];
                    double m_mp = -2.*(m0[i]*mp0[i] + m1[i]*mp1[i] + m2[i]*mp2[i]);
                    double m_m = -(m0[i] * m0[i] + m1[i] * m1[i] + m2[i] * m2[i]);

                    jtimes0[i] = damping_coeff*(mp_h_m_hp*m0[i] + m_h*mp0[i] + m_mp*h0[i] + m_m*hp0[i]);
                    jtimes1[i] = damping_coeff*(mp_h_m_hp*m1[i] + m_h*mp1[i] + m_mp*h1[i] + m_m*hp1[i]);
                    jtimes2[i] = damping_coeff*(mp_h_m_hp*m2[i] + m_h*mp2[i] + m_mp*h2[i] + m_m*hp2[i]);

                    // add relaxation of |m|: (1 - (m.m))m
                    // derivative is -2(m.mp)m + (1-(m.m))mp
                    jtimes0[i] += relax_coeff*(m_mp*m0[i] + (1.+m_m)*mp0[i]);
                    jtimes1[i] += relax_coeff*(m_mp*m1[i] + (1.+m_m)*mp1[i]);
                    jtimes2[i] += relax_coeff*(m_mp*m2[i] + (1.+m_m)*mp2[i]);
                }
            }
        }

        /*
            Computes the solid angle subtended by the triangular mesh Ts, as seen from xs
              r - 3 x m array of points in space
              T - 3 x 3 x n array of triangular coordinates, first index is node number, second index is spatial coordinate
              a (output, optional) -  m vector of computed solid angles

            Return a, or a newly allocated result vector if a is not provided
        */
        np_array<double> compute_solid_angle(const np_array<double> &r_arr, const np_array<double> &T_arr, bp::object a_obj) {
            r_arr.check_ndim(2, "compute_solid_angle: r");
            T_arr.check_ndim(3, "compute_solid_angle: T");
            int m = r_arr.dim()[1];
            int n = T_arr.dim()[2];

            r_arr.check_shape(3, m, "compute_solid_angle: r");
            T_arr.check_shape(3, 3, n, "compute_solid_angle: T");

            np_array<double> a_arr = a_obj.is_none() ? np_array<double>(m) : bp::extract<np_array<double> >(a_obj);
            a_arr.check_shape(m, "compute_solid_angle: a");

            // set up the pointers
            double *r_x = r_arr(0), *r_y = r_arr(1), *r_z = r_arr(2);
            double *T1_x = T_arr(0, 0), *T1_y = T_arr(0, 1), *T1_z = T_arr(0, 2);
            double *T2_x = T_arr(1, 0), *T2_y = T_arr(1, 1), *T2_z = T_arr(1, 2);
            double *T3_x = T_arr(2, 0), *T3_y = T_arr(2, 1), *T3_z = T_arr(2, 2);
            double *a = a_arr.data();

            // i runs over points, j runs over triangles
    //        #pragma omp parallel for schedule(guided)
            for (int i = 0; i < m; i++) {
                double omega = 0;
                for (int j = 0; j < n; j++) {
                    double R1_x = T1_x[j] - r_x[i], R1_y = T1_y[j] - r_y[i], R1_z = T1_z[j] - r_z[i];
                    double R2_x = T2_x[j] - r_x[i], R2_y = T2_y[j] - r_y[i], R2_z = T2_z[j] - r_z[i];
                    double R3_x = T3_x[j] - r_x[i], R3_y = T3_y[j] - r_y[i], R3_z = T3_z[j] - r_z[i];
                    // Wikipedia cites
                    // Van Oosterom, A; Strackee, J (1983). "The Solid Angle of a Plane Triangle". IEEE Trans. Biom. Eng. BME-30 (2): 125–126. doi:10.1109/TBME.1983.325207
                    // Omega = 2*atan(p/q) where
                    // p = R1.R2xR3
                    double p = R1_x*(R2_y*R3_z - R2_z*R3_y) - R2_x*(R1_y*R3_z - R1_z*R3_y) + R3_x*(R1_y*R2_z - R1_z*R2_y);
                    // q = |R1||R2||R3| + |R3|R1.R2 + |R2|R1.R3 + |R1|R2.R3
                    double R1_norm = sqrt(R1_x*R1_x + R1_y*R1_y + R1_z*R1_z);
                    double R2_norm = sqrt(R2_x*R2_x + R2_y*R2_y + R2_z*R2_z);
                    double R3_norm = sqrt(R3_x*R3_x + R3_y*R3_y + R3_z*R3_z);
                    double R1_R2 = R1_x*R2_x + R1_y*R2_y + R1_z*R2_z;
                    double R1_R3 = R1_x*R3_x + R1_y*R3_y + R1_z*R3_z;
                    double R2_R3 = R2_x*R3_x + R2_y*R3_y + R2_z*R3_z;
                    double q = R1_norm*R2_norm*R3_norm + R3_norm*R1_R2 + R2_norm*R1_R3 + R1_norm*R2_R3;

                    double at = atan2(p, q);
                    omega += 2*at;
                }
                a[i] = omega;
            }

            return a_arr;
        }

        std::string demo_hello(std::string name) {
            std::stringstream ss;
            ss << "Hello" << name;
            return ss.str();
        }


	//------------------------------------------------------------------------------------------
	//compute Baryakhtar term for the case that cubic crystal aniostropy
	// dM/dt=-gamma MxH + gamma abs(M) (alpha H - beta Delta H)
        void calc_baryakhtar_dmdt(
                const np_array<double> &M,
                const np_array<double> &H,
		const np_array<double> &delta_H,
                double t,
                const np_array<double> &dmdt,
                const np_array<long> &pins,
                double gamma,
                const np_array<double> &alpha,
		double beta,
                double char_time,
                bool do_precession) {
            M.check_ndim(2, "calc_baryakhtar_dmdt: m");
            int n = M.dim()[1];

            M.check_shape(3, n, "calc_baryakhtar_dmdt: M");
            H.check_shape(3, n, "calc_baryakhtar_dmdt: H");
	    delta_H.check_shape(3, n, "calc_baryakhtar_dmdt: delta_H");
            dmdt.check_shape(3, n, "calc_baryakhtar_dmdt: dmdt");

            alpha.check_ndim(1, "calc_baryakhtar_dmdt: alpha");
            alpha.check_shape(n, "calc_baryakhtar_dmdt: alpha");

            double *m0 = M(0), *m1 = M(1), *m2 = M(2);
            double *h0 = H(0), *h1 = H(1), *h2 = H(2);
	    double *dh0 = delta_H(0), *dh1 = delta_H(1), *dh2 = delta_H(2);
            double *dm0 = dmdt(0), *dm1 = dmdt(1), *dm2 = dmdt(2);
            double tmp_alpha,tmp_beta;
            //double relax_coeff = 0.1/char_time;

            finmag::util::scoped_gil_release release_gil;

            // calculate dmdt
            #pragma omp parallel for schedule(guided)
            for (int i=0; i < n; i++) {

                
                tmp_alpha = (*alpha[i]);
                tmp_beta = beta;

                
                double tmp_coeff = gamma*sqrt(m0[i] * m0[i] + m1[i] * m1[i] + m2[i] * m2[i]);
                dm0[i] = tmp_coeff*(tmp_alpha*h0[i] - tmp_beta*dh0[i]);
                dm1[i] = tmp_coeff*(tmp_alpha*h1[i] - tmp_beta*dh1[i]);
                dm2[i] = tmp_coeff*(tmp_alpha*h2[i] - tmp_beta*dh2[i]);

                
                if (do_precession) {
                    // add precession: - gamma m x H
		    
                    dm0[i] -= gamma*cross0(m0[i], m1[i], m2[i], h0[i], h1[i], h2[i]);
                    dm1[i] -= gamma*cross1(m0[i], m1[i], m2[i], h0[i], h1[i], h2[i]);
                    dm2[i] -= gamma*cross2(m0[i], m1[i], m2[i], h0[i], h1[i], h2[i]);
		    //printf("%g  %g  %g  %g  %g  %g\n",m0[i], m1[i], m2[i],dm0[i],dm1[i],dm2[i]);
                }
            }

            pins.check_ndim(1, "calc_baryakhtar_dmdt: pins");
            const int nb_pins = pins.dim()[0];
            for (int i = 0; i < nb_pins; i++) {
                dm0[*pins[i]] = 0;
                dm1[*pins[i]] = 0;
                dm2[*pins[i]] = 0;
            }
        }

	void calc_baryakhtar_jtimes(
                const np_array<double> &m,
                const np_array<double> &H,
                const np_array<double> &mp,
                const np_array<double> &Hp,
                double t,
                const np_array<double> &jtimes,
                double gamma_LL,
                double alpha,
                double char_time,
                bool do_precession) {

            m.check_ndim(2, "calc_llg_dmdt: m");
            int n = m.dim()[1];

            m.check_shape(3, n, "calc_llg_dmdt: m");
            mp.check_shape(3, n, "calc_llg_dmdt: mp");
            H.check_shape(3, n, "calc_llg_dmdt: H");
            Hp.check_shape(3, n, "calc_llg_dmdt: Hp");
            jtimes.check_shape(3, n, "calc_llg_dmdt: jtimes");

            double precession_coeff = -gamma_LL;
            double damping_coeff = -gamma_LL*alpha;
            double *m0 = m(0), *m1 = m(1), *m2 = m(2);
            double *mp0 = mp(0), *mp1 = mp(1), *mp2 = mp(2);
            double *h0 = H(0), *h1 = H(1), *h2 = H(2);
            double *hp0 = Hp(0), *hp1 = Hp(1), *hp2 = Hp(2);
            double *jtimes0 = jtimes(0), *jtimes1 = jtimes(1), *jtimes2 = jtimes(2);
            double relax_coeff = 0.1/char_time;

            finmag::util::scoped_gil_release release_gil;

            if (do_precession) {
                #pragma omp parallel for schedule(guided)
                for (int i = 0; i < n; i++) {
                    // add precession: mp x H + m x Hp
                    jtimes0[i] = precession_coeff*(cross0(mp0[i], mp1[i], mp2[i], h0[i], h1[i], h2[i])
                            + cross0(m0[i], m1[i], m2[i], hp0[i], hp1[i], hp2[i]));
                    jtimes1[i] = precession_coeff*(cross1(mp0[i], mp1[i], mp2[i], h0[i], h1[i], h2[i])
                            + cross1(m0[i], m1[i], m2[i], hp0[i], hp1[i], hp2[i]));
                    jtimes2[i] = precession_coeff*(cross2(mp0[i], mp1[i], mp2[i], h0[i], h1[i], h2[i])
                            + cross2(m0[i], m1[i], m2[i], hp0[i], hp1[i], hp2[i]));

                    // add damping: m x (m x H) == (m.H)m - (m.m)H, multiplied by -gamma alpha
                    // derivative is [(mp.H) + (m.Hp)]m + (m.H)mp - 2(m.mp)H - (m.m)Hp
                    double mp_h_m_hp = mp0[i]*h0[i] + mp1[i]*h1[i] + mp2[i]*h2[i]
                            + m0[i]*hp0[i] + m1[i]*hp1[i] + m2[i]*hp2[i];
                    double m_h = m0[i] * h0[i] + m1[i] * h1[i] + m2[i] * h2[i];
                    double m_mp = -2.*(m0[i]*mp0[i] + m1[i]*mp1[i] + m2[i]*mp2[i]);
                    double m_m = -(m0[i] * m0[i] + m1[i] * m1[i] + m2[i] * m2[i]);

                    jtimes0[i] += damping_coeff*(mp_h_m_hp*m0[i] + m_h*mp0[i] + m_mp*h0[i] + m_m*hp0[i]);
                    jtimes1[i] += damping_coeff*(mp_h_m_hp*m1[i] + m_h*mp1[i] + m_mp*h1[i] + m_m*hp1[i]);
                    jtimes2[i] += damping_coeff*(mp_h_m_hp*m2[i] + m_h*mp2[i] + m_mp*h2[i] + m_m*hp2[i]);

                    // add relaxation of |m|: (1 - (m.m))m
                    // derivative is -2(m.mp)m + (1-(m.m))mp
                   // jtimes0[i] += relax_coeff*(m_mp*m0[i] + (1.+m_m)*mp0[i]);
                   // jtimes1[i] += relax_coeff*(m_mp*m1[i] + (1.+m_m)*mp1[i]);
                   // jtimes2[i] += relax_coeff*(m_mp*m2[i] + (1.+m_m)*mp2[i]);
                }
            } else {
                #pragma omp parallel for schedule(guided)
                for (int i = 0; i < n; i++) {
                    // add damping: m x (m x H) == (m.H)m - (m.m)H, multiplied by -gamma alpha
                    // derivative is [(mp.H) + (m.Hp)]m + (m.H)mp - 2(m.mp)H - (m.m)Hp
                    double mp_h_m_hp = mp0[i]*h0[i] + mp1[i]*h1[i] + mp2[i]*h2[i]
                            + m0[i]*hp0[i] + m1[i]*hp1[i] + m2[i]*hp2[i];
                    double m_h = m0[i] * h0[i] + m1[i] * h1[i] + m2[i] * h2[i];
                    double m_mp = -2.*(m0[i]*mp0[i] + m1[i]*mp1[i] + m2[i]*mp2[i]);
                    double m_m = -(m0[i] * m0[i] + m1[i] * m1[i] + m2[i] * m2[i]);

                    jtimes0[i] = damping_coeff*(mp_h_m_hp*m0[i] + m_h*mp0[i] + m_mp*h0[i] + m_m*hp0[i]);
                    jtimes1[i] = damping_coeff*(mp_h_m_hp*m1[i] + m_h*mp1[i] + m_mp*h1[i] + m_m*hp1[i]);
                    jtimes2[i] = damping_coeff*(mp_h_m_hp*m2[i] + m_h*mp2[i] + m_mp*h2[i] + m_m*hp2[i]);

                    // add relaxation of |m|: (1 - (m.m))m
                    // derivative is -2(m.mp)m + (1-(m.m))mp
                    jtimes0[i] += relax_coeff*(m_mp*m0[i] + (1.+m_m)*mp0[i]);
                    //jtimes1[i] += relax_coeff*(m_mp*m1[i] + (1.+m_m)*mp1[i]);
                    //jtimes2[i] += relax_coeff*(m_mp*m2[i] + (1.+m_m)*mp2[i]);
                }
            }
        }


    }

    void register_llg() {
        using namespace bp;

        def("calc_llg_dmdt", &calc_llg_dmdt, (
            arg("m"),
            arg("H"),
            arg("t"),
            arg("dmdt"),
            arg("pins"),
            arg("gamma_LL"),
            arg("alpha"),
            arg("char_time"),
            arg("do_precession")
        ));
        def("calc_llg_jtimes", &calc_llg_jtimes, (
            arg("m"),
            arg("H"),
            arg("mp"),
            arg("Hp"),
            arg("t"),
            arg("jtimes"),
            arg("gamma_LL"),
            arg("alpha"),
            arg("char_time"),
            arg("do_precession")
        ));
        def("compute_solid_angle", &compute_solid_angle, (
            arg("r"),
            arg("T"),
            arg("a")=object()
        ));
        //Towards some simple examples on using Boost.Python:
        def("demo_hello", &demo_hello, (
                arg("name")
                    ));

	def("calc_baryakhtar_dmdt", &calc_baryakhtar_dmdt, (
            arg("M"),
            arg("H"),
	    arg("delta_H"),
            arg("t"),
            arg("dmdt"),
            arg("pins"),
            arg("gamma_LL"),
            arg("alpha"),
	    arg("beta"),
            arg("char_time"),
            arg("do_precession")
        ));

        def("calc_baryakhtar_jtimes", &calc_baryakhtar_jtimes, (
            arg("m"),
            arg("H"),
            arg("mp"),
            arg("Hp"),
            arg("t"),
            arg("jtimes"),
            arg("gamma_LL"),
            arg("alpha"),
            arg("char_time"),
            arg("do_precession")
        ));
    }
}}
