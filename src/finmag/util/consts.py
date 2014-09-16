"""
A collection of constants we use (in SI units).

"""
from __future__ import division
from math import pi, sqrt

mu0 = 4 * pi * 1e-7  # Vs/(Am)
k_B = 1.3806488e-23  # Boltzmann constant in J / K
h_bar = 1.054571726e-34  # reduced Plank constant in Js
e = 1.602176565e-19  # elementary charge in As
# m/(As) (source:  OOMMF manual, and in Werner Scholz thesis, after (3.7),
# llg_gamma_G = m/(As))
gamma = 2.210173e5
ONE_DEGREE_PER_NS = 17453292.52  # rad/s


def exchange_length(A, Ms):
    """
    Computes the exchange length, when given the exchange constant A
    and the saturation magnetisation Ms.

    """
    return sqrt(2 * A / (mu0 * Ms ** 2))


def bloch_parameter(A, K1):
    """
    Computes the Bloch parameter, when given the exchange constant A
    and the anisotropy energy density K1.

    """
    return sqrt(A / K1)


def helical_period(A, D):
    """
    Computes the Helical period of a Skyrmion, when given exchange
    constant, A and the DMI strength,D. 
    """
    return 4 * pi * A / abs(D)


def flux_density_to_field_strength(B):
    """
    Converts the magnetic flux density to the magnetic field strength.

    Magnetic flux density B is expressed in Tesla, and the returned field
    strength H is expressed in A/m.

    """
    H = B / mu0
    return H


def Oersted_to_SI(H):
    """
    Converts the magnetic field strength H from Oersted to A/m.

    """
    return H * 1e3 / (4 * pi)
