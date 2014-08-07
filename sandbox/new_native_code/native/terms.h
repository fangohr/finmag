#pragma once
#include <dolfin/function/Function.h>

namespace dolfin { namespace finmag {
    void damping(double const& alpha, double const& gamma,
                 double const& m_x, double const& m_y, double const& m_z,
                 double const& H_x, double const& H_y, double const& H_z,
                 double& dm_x, double& dm_y, double& dm_z);

    void precession(double const& alpha, double const& gamma,
                    double const& m_x, double const& m_y, double const& m_z,
                    double const& H_x, double const& H_y, double const& H_z,
                    double& dm_x, double& dm_y, double& dm_z);

    void relaxation(double const& c,
                    double const& m_x, double const& m_y, double const& m_z,
                    double& dm_x, double& dm_y, double& dm_z);

    class Slonczewski {
        public:
            Slonczewski(double const d, double const P, Array<double> const& p,
                        double const lambda, double const epsilonprime);
            void compute(double const& alpha, double const& gamma,
                         double const& J, double const& Ms,
                         double const& m_x, double const& m_y, double const& m_z,
                         double& dm_x, double& dm_y, double& dm_z);
        private:
            double lambda;
            double P; /* degree of polarisation */
            double d; /* thickness of free layer in SI units */
            double p_x, p_y, p_z; /* fixed layer magnetisation direction */
            double epsilonprime; /* secondary spin-transfer term */
    };

    class ZhangLi {
        public:
            ZhangLi(double const u_0, double const beta);
            void compute(double const& alpha, double const& Ms,
                         double const& m_x, double const& m_y, double const& m_z,
                         double const& g_x, double const& g_y, double const& g_z,
                         double& dm_x, double& dm_y, double& dm_z);
        private:
            double u_0;
            double beta;
    };

    class NonlocalSTT {
        public:
            NonlocalSTT(double const P, double const tau_sd, double const tau_sf);
            void compute(double const& alpha, double const& Ms,
                         double const& m_x, double const& m_y, double const& m_z,
                         double const& g_x, double const& g_y, double const& g_z,
                         double const& d_x, double const& d_y, double const& d_z,
                         double& dm_x, double& dm_y, double& dm_z);
        private:
            double P;
            double tau_sd;
            double tau_sf;
    };
}}
