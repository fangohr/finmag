from math import pi, sqrt
from finmag.util.consts import mu0, gamma

# my guess (Permalloy), paper uses D/(gamma*h_bar), no numerical value
A = 13.0e-12

l_ex_ref = 6e-9
larmor_frequency_ref = 2 * pi * 23e9

for Ms in [640e3, 860e3]: # Sec. 3 p. 4 and Sec. 5, p. 8.
    print "--- With Ms {:.2} ---".format(Ms)

    # Should be around 6 nm, Sec. 3 p. 4.
    l_ex = sqrt(2 * A / (mu0 * Ms**2))
    print "The exchange length is l_ex = {:.2} m, given as {:.2}.".format(l_ex, l_ex_ref)

    # Should be 2 pi * 23 GHz, Sec. 3 p. 4.
    larmor_frequency = 2 * pi * gamma * mu0 * Ms
    print "Larmor frequency is {:.2}, given as {:.2}.".format(larmor_frequency, larmor_frequency_ref)

    time_unit = larmor_frequency / (2 * pi)
    print "1 time unit in the paper is {:.2}s, could be 50 ps.".format(time_unit)
    
    print ""
