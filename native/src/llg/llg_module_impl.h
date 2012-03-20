#pragma once

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
        int n = m.dims()[1];

        m.check_shape(3, n, "calc_llg_dmdt: m");
        H.check_shape(3, n, "calc_llg_dmdt: H");
        dmdt.check_shape(3, n, "calc_llg_dmdt: H");

        double precession_coeff = -gamma_LL;
        double damping_coeff = -gamma_LL*alpha;
        double *m0 = m(0), *m1 = m(1), *m2 = m(2);
        double *h0 = H(0), *h1 = H(1), *h2 = H(2);
        double *dm0 = dmdt(0), *dm1 = dmdt(1), *dm2 = dmdt(2);
        double relax_coeff = 0.1/char_time;

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

    void calc_jtimes(
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
        int n = m.dims()[1];

        m.check_shape(3, n, "calc_llg_dmdt: m");
        H.check_shape(3, n, "calc_llg_dmdt: H");
        dmdt.check_shape(3, n, "calc_llg_dmdt: H");

        double precession_coeff = -gamma_LL;
        double damping_coeff = -gamma_LL*alpha;
        double *m0 = m(0), *m1 = m(1), *m2 = m(2);
        double *h0 = H(0), *h1 = H(1), *h2 = H(2);
        double *jtimes0 = jtimes(0), *jtimes1 = jtimes(1), *jtimes2 = jtimes(2);
        double relax_coeff = 0.1/char_time;

        if (do_precession) {
            #pragma omp parallel for schedule(guided)
            for (int i = 0; i < n; i++) {
                // add precession: mp x H + m x Hp
                jtimes[i] +=
                // add damping: M x (M x H) == (M.H)M - H, multiplied by -gamma
                // add relaxation of |m|
            }
        } else {
            #pragma omp parallel for schedule(guided)
            for (int i = 0; i < n; i++) {
                // add damping: M x (M x H) == (M.H)M - H, multiplied by -gamma
                // add relaxation of |m|
            }
        }
    }

    void init_module() {
        bp::def("calc_llg_dmdt", &calc_llg_dmdt);
    }
}}