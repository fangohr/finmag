#pragma once

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

    void slonczewski(double const& alpha, double const& gamma,
                     double const& lambda, double const& epsilonprime,
                     double const& J, double const& P, double const& d, double const& Ms,
                     double const& m_x, double const& m_y, double const& m_z,
                     double const& p_x, double const& p_y, double const& p_z,
                     double& dm_x, double& dm_y, double& dm_z);
}}
