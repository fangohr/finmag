#pragma once

#include "util/python_threading.h"

namespace finmag { namespace llg {
    // Components of a cross product
    inline double cross0(double a0, double a1, double a2, double b0, double b1, double b2) { return a1*b2 - a2*b1; }
    inline double cross1(double a0, double a1, double a2, double b0, double b1, double b2) { return a2*b0 - a0*b2; }
    inline double cross2(double a0, double a1, double a2, double b0, double b1, double b2) { return a0*b1 - a1*b0; }

    void calc_llg_dmdt(
            const np_array<double> &m,
            const np_array<double> &H,
            double t,
            const np_array<double> &dmdt,
            double gamma_LL,
            double alpha,
            double char_time,
            bool do_precession) {
        m.check_ndim(2, "calc_llg_dmdt: m");
        int n = m.dim()[1];

        m.check_shape(3, n, "calc_llg_dmdt: m");
        H.check_shape(3, n, "calc_llg_dmdt: H");
        dmdt.check_shape(3, n, "calc_llg_dmdt: dmdt");

        double precession_coeff = -gamma_LL;
        double damping_coeff = -gamma_LL*alpha;
        double *m0 = m(0), *m1 = m(1), *m2 = m(2);
        double *h0 = H(0), *h1 = H(1), *h2 = H(2);
        double *dm0 = dmdt(0), *dm1 = dmdt(1), *dm2 = dmdt(2);
        double relax_coeff = 0.1/char_time;

        finmag::util::scoped_gil_release release_gil;

        // calculate dm
        if (do_precession) {
            #pragma omp parallel for schedule(guided)
            for (int i = 0; i < n; i++) {
                // add precession: - gamma m x H
                dm0[i] = precession_coeff*cross0(m0[i], m1[i], m2[i], h0[i], h1[i], h2[i]);
                dm1[i] = precession_coeff*cross1(m0[i], m1[i], m2[i], h0[i], h1[i], h2[i]);
                dm2[i] = precession_coeff*cross2(m0[i], m1[i], m2[i], h0[i], h1[i], h2[i]);

                // add damping: m x (m x H) == (m.H)m - (m.m)H, multiplied by -gamma alpha
                double mh = m0[i] * h0[i] + m1[i] * h1[i] + m2[i] * h2[i];
                double mm = m0[i] * m0[i] + m1[i] * m1[i] + m2[i] * m2[i];
                dm0[i] += damping_coeff*(m0[i] * mh - h0[i] * mm);
                dm1[i] += damping_coeff*(m1[i] * mh - h1[i] * mm);
                dm2[i] += damping_coeff*(m2[i] * mh - h2[i] * mm);

                // add relaxation of |m|
                double relax = relax_coeff*(1.-mm);
                dm0[i] += relax*m0[i];
                dm1[i] += relax*m1[i];
                dm2[i] += relax*m2[i];
            }
        } else {
            #pragma omp parallel for schedule(guided)
            for (int i = 0; i < n; i++) {
                // add precession: - gamma m x H
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
            }
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

    void register_module() {
        using namespace bp;

        def("calc_llg_dmdt", &calc_llg_dmdt, (
            arg("m"),
            arg("H"),
            arg("t"),
            arg("dmdt"),
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
    }
}}