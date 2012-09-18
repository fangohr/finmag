from finmag.util.consts import mu0, h_bar
from math import sqrt, pi

"""
Values given in the paper

"""

alpha = 0.012
Ms = 860e3; Ms_alt = 640e3; # A/m

larmor_freq = 2 * pi * 23e9 # Hz
_larmor_freq = lambda Ms: 2 * pi * gamma_LLG * mu0 * Ms

l_ex = 6e-9 # nm exchange length

"""
Additional values, from us

"""

gamma = 2.210173e5 # m/(As) gyromagnetic ratio
gamma_LLG = gamma / (1 + alpha ** 2)
A = 13.0e-12 # exchange constant for Permalloy in J/m

l_ex_A = sqrt(2 * A / (mu0 * Ms ** 2))

# their exchange constant, no value in paper, formula by comparison with known formulas
D = 2 * A * gamma_LLG * h_bar / Ms
# their exchange length formula
l_ex_D = sqrt(D / (gamma_LLG * mu0 * Ms * h_bar))

time = lambda time_units, w_M: time_units * 2 * pi / w_M 

"""
Comparison

"""

print "EXCHANGE LENGTH"
print "Value for exchange length given in the paper l_ex= {} m.".format(l_ex)
print "When computed for Permalloy l_ex= {:.2g} m.\n".format(l_ex_A)
assert abs(l_ex_A - l_ex_D) < 1e-15

print "EXCHANGE CONSTANT"
print "Exchange constant for Permalloy we use A= {:.2g} J/m.".format(A)
A_from_l_ex = l_ex ** 2 * mu0 * Ms**2 / 2
print "Their value must have been A= {:.2g} J/m.\n".format(A_from_l_ex)

print "LARMOR FREQUENCY"
print "Value for Larmor frequency given in the paper w_M= 2 * pi * 23GHz = {:.2g} Hz.".format(larmor_freq)
print "1. When computed with the first value of Ms in paper, Ms= {:.2g} A/m:".format(Ms_alt)
print "   w_M= 2 * pi * {:.2g} MHz = {:.2g} Hz.".format(_larmor_freq(Ms_alt)/(2*pi*1e6),_larmor_freq(Ms_alt))
print "2. When computed with Ms= {:.2g} A/m:".format(Ms)
print "   w_M= 2 * pi * {:.2g} MHz = {:.2g} Hz.".format(_larmor_freq(Ms)/(2*pi*1e6),_larmor_freq(Ms))
print "   The digits come close, but the magnitude is wrong. Probably an oversight in the paper.\n"

print "TIME"
print "Paper uses dimensionless time based on the Larmor frequency."
print "Value for one time unit given in text, around 50 ps."
print "1. When computed with their (wrong?) value of the Larmor frequency."
print "   {} time unit = {:.2g} ps. Consistent with value in paper.".format(1, 1e12*time(1, larmor_freq))
print "2. When computed with the Larmor frequency for Ms= {:.2g} A/m.".format(Ms)
print "   {} time unit = {:.2g} microseconds.".format(1, 1e6*time(1, _larmor_freq(Ms)))
print "3. When computed with the Larmor frequency for Ms= {:.2g} A/m.".format(Ms_alt)
print "   {} time unit = {:.2g} microseconds.".format(1, 1e6*time(1, _larmor_freq(Ms_alt)))

