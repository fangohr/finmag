import numpy as np


def macrospin_analytic_solution(alpha, gamma, H, t_array):
    """
    Computes the analytic solution of magnetisation x component
    as a function of time for the macrospin in applied external
    magnetic field H.

    Source: PhD Thesis Matteo Franchin,
    http://eprints.soton.ac.uk/161207/1.hasCoversheetVersion/thesis.pdf,
    Appendix B, page 127

    """
    t0 = 1 / (gamma * alpha * H) * \
        np.log(np.sin(np.pi / 2) / (1 + np.cos(np.pi / 2)))
    mx_analytic = []
    for t in t_array:
        phi = gamma * H * t                                     # (B16)
        costheta = np.tanh(gamma * alpha * H * (t - t0))        # (B17)
        sintheta = 1 / np.cosh(gamma * alpha * H * (t - t0))    # (B18)
        mx_analytic.append(sintheta * np.cos(phi))

    return np.array(mx_analytic)
