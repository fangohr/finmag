/* Compute the damping for one node. */
void damping(double const& alpha, double const& gamma,
             double const& m0, double const& m1, double const& m2,
             double const& H0, double const& H1, double const& H2,
             double &dm0, double &dm1, double &dm2) {
    double const damping_prefactor = - alpha * gamma;
    /* vector triple product: m x (m x H) = m(m*H) - H(m*m) */
    double const mH = m0 * H0 + m1 * H1 + m2 * H2;
    double const mm = m0 * m0 + m1 * m1 + m2 * m2;
    dm0 += damping_prefactor * (m0 * mH - H0 * mm); 
    dm1 += damping_prefactor * (m1 * mH - H1 * mm); 
    dm2 += damping_prefactor * (m2 * mH - H2 * mm); 
}
