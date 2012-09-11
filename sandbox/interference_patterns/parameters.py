from finmag.util.consts import mu0, h_bar, gamma
from math import sqrt, pi

epsilon = 1e-14

alpha = 0.012
gamma_LLG = gamma / (1 + alpha ** 2)

Ms = 640e3

# exchange constant for Permalloy in J/m
A = 13.0e-12
# exchange constant used in the paper without value.
# formula obtained by comparing eq. 2 and exchange length formula in paper
# to usual exchange length and exchange field computation.
D = 2 * A * gamma_LLG * h_bar / Ms

l_ex = sqrt(2 * A / (mu0 * Ms ** 2))
l_ex_paper = sqrt(D / (gamma_LLG * mu0 * Ms * h_bar))
assert abs(l_ex - l_ex_paper) < epsilon

larmor_frequency = 2 * pi * gamma_LLG * mu0 * Ms
larmor_frequency_paper = 2 * pi * 23e9

if __name__ == "__main__":
    print "The exchange length is l_ex = {:.2} m.".format(l_ex)
    print "Larmor frequency is {:.2g} Hz, expected {:.2g} Hz.".format(
            larmor_frequency, larmor_frequency_paper)
