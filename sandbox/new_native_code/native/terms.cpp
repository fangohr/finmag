#include "terms.h"

/* Compute the damping for one node. */
void damping(double const& alpha, double const& gamma,
             double const& m_x, double const& m_y, double const& m_z,
             double const& H_x, double const& H_y, double const& H_z,
             double& dm_x, double& dm_y, double& dm_z) {
    double const damping_prefactor = - alpha * gamma / (1 + alpha * alpha);
    /* vector triple product: m x (m x H) = m(m*H) - H(m*m) */
    double const mH = m_x * H_x + m_y * H_y + m_z * H_z;
    double const mm = m_x * m_x + m_y * m_y + m_z * m_z;
    dm_x += damping_prefactor * (m_x * mH - H_x * mm); 
    dm_y += damping_prefactor * (m_y * mH - H_y * mm); 
    dm_z += damping_prefactor * (m_z * mH - H_z * mm); 
}
