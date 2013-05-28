import numpy
import numbers


def make_analytic_solution(H, alpha, gamma):
    """
    Returns a function `m(t)` which computes the magnetisation vector
    of a macrospin as a function of time, i.e. the typical precession
    under the influence of an applied field `H`. Assumes the initial
    condition m(0) = (1,0,0).

    Arguments:

        - H: the magnitude of the applied field (in A/m)

        - alpha: Gilbert damping factor (dimensionless)

        - gamma (alias OOMMF's gamma_G): gyromagnetic ratio (in m/A*s)

    """
    if not isinstance(H, numbers.Number):
        raise TypeError("H must be a single number, but got: {}".format(H))
    p = float(gamma) / (1 + alpha ** 2)
    theta0 = numpy.pi / 2
    t0 = numpy.log(numpy.sin(theta0) / (1 + numpy.cos(theta0))) / (p * alpha * H)

    # Matteo uses spherical coordinates,
    # which have to be converted to cartesian coordinates.

    def phi(t):
        return p * H * t

    def cos_theta(t):
        return numpy.tanh(p * alpha * H * (t - t0))

    def sin_theta(t):
        return 1 / (numpy.cosh(p * alpha * H * (t - t0)))

    def x(t):
        return sin_theta(t) * numpy.cos(phi(t))

    def y(t):
        return sin_theta(t) * numpy.sin(phi(t))

    def z(t):
        return cos_theta(t)

    def m(t):
        return numpy.array([x(t), y(t), z(t)])

    return m
