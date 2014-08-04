#include "terms.h"

namespace dolfin { namespace finmag { 
    /* Components of the cross product. */
    inline double cross_x(double ax, double ay, double az, double bx, double by, double bz) { return ay*bz - az*by; }
    inline double cross_y(double ax, double ay, double az, double bx, double by, double bz) { return az*bx - ax*bz; }
    inline double cross_z(double ax, double ay, double az, double bx, double by, double bz) { return ax*by - ay*bx; }

    /* Compute the damping for one node. */
    void damping(double const& alpha, double const& gamma,
                 double const& m_x, double const& m_y, double const& m_z,
                 double const& H_x, double const& H_y, double const& H_z,
                 double& dm_x, double& dm_y, double& dm_z) {
        double const prefactor = - alpha * gamma / (1 + alpha * alpha);
        /* vector triple product: m x (m x H) = m(m*H) - H(m*m) */
        double const mH = m_x * H_x + m_y * H_y + m_z * H_z;
        double const mm = m_x * m_x + m_y * m_y + m_z * m_z;
        /* overwrites dm_x, dm_y, dm_z and thus always needs to be computed first */
        dm_x = prefactor * (m_x * mH - H_x * mm); 
        dm_y = prefactor * (m_y * mH - H_y * mm); 
        dm_z = prefactor * (m_z * mH - H_z * mm); 
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
}}
