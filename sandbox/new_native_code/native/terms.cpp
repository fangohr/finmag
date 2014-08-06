#include <math.h>
#include "terms.h"

namespace dolfin { namespace finmag { 
    /* Components of the cross product. */
    inline double cross_x(double ax, double ay, double az, double bx, double by, double bz) { return ay*bz - az*by; }
    inline double cross_y(double ax, double ay, double az, double bx, double by, double bz) { return az*bx - ax*bz; }
    inline double cross_z(double ax, double ay, double az, double bx, double by, double bz) { return ax*by - ay*bx; }

    double const e = 1.602176565e-19; // elementary charge in As
    double const h_bar = 1.054571726e-34; // reduced Plank constant in Js
    double const pi = 4 * atan(1);
    double const mu_B = 9.27400968e-24; //Bohr magneton
    double const mu_0 = pi * 4e-7; // Vacuum permeability in Vs/(Am)

    /* Compute the damping for one node. */
    void damping(double const& alpha, double const& gamma,
                 double const& m_x, double const& m_y, double const& m_z,
                 double const& H_x, double const& H_y, double const& H_z,
                 double& dm_x, double& dm_y, double& dm_z) {
        double const prefactor = - alpha * gamma / (1 + alpha * alpha);
        /* vector triple product: m x (m x H) = m(m*H) - H(m*m) */
        double const mH = m_x * H_x + m_y * H_y + m_z * H_z;
        double const mm = m_x * m_x + m_y * m_y + m_z * m_z;
        dm_x += prefactor * (m_x * mH - H_x * mm); 
        dm_y += prefactor * (m_y * mH - H_y * mm); 
        dm_z += prefactor * (m_z * mH - H_z * mm); 
    }

    /* Compute the precession for one node. */
    void precession(double const& alpha, double const& gamma,
                    double const& m_x, double const& m_y, double const& m_z,
                    double const& H_x, double const& H_y, double const& H_z,
                    double& dm_x, double& dm_y, double& dm_z) {
        double const prefactor = - gamma / (1 + alpha * alpha);
        dm_x += prefactor * cross_x(m_x, m_y, m_z, H_x, H_y, H_z);
        dm_y += prefactor * cross_y(m_x, m_y, m_z, H_x, H_y, H_z);
        dm_z += prefactor * cross_z(m_x, m_y, m_z, H_x, H_y, H_z);
    }

    /* Compute the relaxation for one node. */
    void relaxation(double const& c,
                    double const& m_x, double const& m_y, double const& m_z,
                    double& dm_x, double& dm_y, double& dm_z) {
        double const mm = m_x * m_x + m_y * m_y + m_z * m_z;
        double const prefactor = c * (1.0 - mm); 
        dm_x += prefactor * m_x;
        dm_y += prefactor * m_y;
        dm_z += prefactor * m_z;
    }

    /* Compute the Slonczewski/Xiao spin-torque term for one node. */
    Slonczewski::Slonczewski(double const d, double const P, Array<double> const& p,
                             double const lambda, double const epsilonprime) :
            d(d),
            P(P),
            p_x(p[0]), p_y(p[1]), p_z(p[2]),
            lambda(lambda),
            epsilonprime(epsilonprime) {
    }

    void Slonczewski::compute(double const& alpha, double const& gamma,
                            double const& J, double const& Ms,
                            double const& m_x, double const& m_y, double const& m_z,
                            double& dm_x, double& dm_y, double& dm_z) {
        double const mm = m_x * m_x + m_y * m_y + m_z * m_z; /* for the vector triple product expansion */
        double const mp = m_x * p_x + m_y * p_y + m_z * p_z; /* also known as Lagrange's formula */

        /* m, J, alpha and Ms may vary over the nodes, leaving us no other choice than
         * to compute the following quantities here, and not once before iteration */
        double const gamma_LL = gamma / (1 + alpha * alpha);
        double const lambda_sq = lambda * lambda;
        double const beta = J * h_bar / (mu_0 * Ms * e * d);
        double const epsilon = P * lambda_sq / (lambda_sq + 1 + (lambda_sq - 1) * mp);

        /* gamma_LL * beta * {(alpha*eps - eps') m x p - (eps - alpha*eps') m x (m x p)} */
        double const perp = alpha * epsilon - epsilonprime;
        double const para = epsilon - alpha * epsilonprime;
        dm_x += gamma_LL * beta * (perp * cross_x(m_x, m_y, m_z, p_x, p_y, p_z) - para * (mp * m_x - mm * p_x));
        dm_y += gamma_LL * beta * (perp * cross_y(m_x, m_y, m_z, p_x, p_y, p_z) - para * (mp * m_y - mm * p_y));
        dm_z += gamma_LL * beta * (perp * cross_z(m_x, m_y, m_z, p_x, p_y, p_z) - para * (mp * m_z - mm * p_z));
    }
}}
