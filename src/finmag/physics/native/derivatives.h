#pragma once

namespace dolfin { namespace finmag {
    void dm_damping(double const& alpha, double const& gamma,
                    double const& m_x,   double const& m_y,  double const& m_z,
                    double const& mp_x,  double const& mp_y, double const& mp_z,
                    double const& H_x,   double const& H_y,  double const& H_z,
                    double const& Hp_x,  double const& Hp_y, double const& Hp_z,
                    double& jtimes_x,    double& jtimes_y,   double& jtimes_z);

    void dm_precession(double const& alpha, double const& gamma,
                       double const& m_x,   double const& m_y,  double const& m_z,
                       double const& mp_x,  double const& mp_y, double const& mp_z,
                       double const& H_x,   double const& H_y,  double const& H_z,
                       double const& Hp_x,  double const& Hp_y, double const& Hp_z,
                       double& jtimes_x,    double& jtimes_y,   double& jtimes_z);

    void dm_relaxation(double const& c,
                       double const& m_x,   double const& m_y,  double const& m_z,
                       double const& mp_x,  double const& mp_y, double const& mp_z,
                       double& jtimes_x,    double& jtimes_y,   double& jtimes_z);
}}
