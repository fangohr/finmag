#include "derivatives.h"

namespace dolfin { namespace finmag {
    /* Components of the cross product. */
    inline double cross_x(double ax, double ay, double az, double bx, double by, double bz) { return ay*bz - az*by; }
    inline double cross_y(double ax, double ay, double az, double bx, double by, double bz) { return az*bx - ax*bz; }
    inline double cross_z(double ax, double ay, double az, double bx, double by, double bz) { return ax*by - ay*bx; }

    /* Compute the derivative of the damping term for one node. */
    void dm_damping(double const& alpha, double const& gamma,
                    double const& m_x,   double const& m_y,  double const& m_z,
                    double const& mp_x,  double const& mp_y, double const& mp_z,
                    double const& H_x,   double const& H_y,  double const& H_z,
                    double const& Hp_x,  double const& Hp_y, double const& Hp_z,
                    double& jtimes_x,    double& jtimes_y,   double& jtimes_z) {
        double const prefactor = - alpha * gamma / (1 + alpha * alpha);
        /* damping: m x (m x H) = m(m*H) - H(m*m)
         * derivative: m'(m*H) + m(m'*H + m*H') - H'(m*m) - H(2*m*m') */
        double const mH = m_x * H_x + m_y * H_y + m_z * H_z;
        double const mpH_mHp = mp_x * H_x + mp_y * H_y + mp_z * H_z + m_x * Hp_x + m_y * Hp_y + m_z * Hp_z;
        double const mm = m_x * m_x + m_y * m_y + m_z * m_z;
        double const mmp = m_x * mp_x + m_y * mp_y + m_z * mp_z;
        jtimes_x += prefactor * (mp_x * mH + m_x * mpH_mHp - Hp_x * mm - H_x * 2 * mmp);
        jtimes_y += prefactor * (mp_y * mH + m_y * mpH_mHp - Hp_y * mm - H_y * 2 * mmp);
        jtimes_z += prefactor * (mp_z * mH + m_z * mpH_mHp - Hp_z * mm - H_z * 2 * mmp);
    }

    /* Compute the derivative of the precessional term for one node. */
    void dm_precession(double const& alpha, double const& gamma,
                       double const& m_x,   double const& m_y,  double const& m_z,
                       double const& mp_x,  double const& mp_y, double const& mp_z,
                       double const& H_x,   double const& H_y,  double const& H_z,
                       double const& Hp_x,  double const& Hp_y, double const& Hp_z,
                       double& jtimes_x,    double& jtimes_y,   double& jtimes_z) {
        double const prefactor = - gamma / (1 + alpha * alpha);
        /* precession: m x H
         * derivative: m' x H + m x H' */
        jtimes_x += prefactor * (cross_x(mp_x, mp_y, mp_z, H_x, H_y, H_z) + cross_x(m_x, m_y, m_z, Hp_x, Hp_y, Hp_z));
        jtimes_y += prefactor * (cross_y(mp_x, mp_y, mp_z, H_x, H_y, H_z) + cross_y(m_x, m_y, m_z, Hp_x, Hp_y, Hp_z));
        jtimes_z += prefactor * (cross_z(mp_x, mp_y, mp_z, H_x, H_y, H_z) + cross_z(m_x, m_y, m_z, Hp_x, Hp_y, Hp_z));
    }

    /* Compute the derivative of the relaxation term for one node. */
    void dm_relaxation(double const& c,
                       double const& m_x,   double const& m_y,  double const& m_z,
                       double const& mp_x,  double const& mp_y, double const& mp_z,
                       double& jtimes_x,    double& jtimes_y,   double& jtimes_z) {
        /* relaxation: (1 - m*m) * m
         * derivative: - 2 * m*m' * m + (1 - m*m) * m' */
        double const mm = m_x * m_x + m_y * m_y + m_z * m_z;
        double const mmp = m_x * mp_x + m_y * mp_y + m_z * mp_z;
        jtimes_x += c * (-2 * mmp * m_x + (1.0 - mm) * mp_x);
        jtimes_y += c * (-2 * mmp * m_y + (1.0 - mm) * mp_y);
        jtimes_z += c * (-2 * mmp * m_z + (1.0 - mm) * mp_z);
    }
}}
