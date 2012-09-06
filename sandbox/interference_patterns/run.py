import dolfin as df
import numpy as np
from math import pi, sin, cos, sqrt, floor
from finmag import Simulation
from finmag.util.consts import mu0
from finmag.util.meshes import from_geofile
from finmag.energies import Zeeman, Exchange, ThinFilmDemag
from point_contacts import point_contacts as point_contacts_gen

# The film described in figures 5 and 6, page 7 and section 5, page 8 is
# 440 nm * 440 nm * 5 nm. We'll put the center of the film at 0,0,0. However,
# the film doesn't have an area of 600 nm^2, as stated in Sec. 5 p. 8.
L = 440.0e-9; W = 440.0e-9; H = 5.0e-9;
dL = L/2; dW = W/2; dH = H/2;
discretisation = 5.0e-9
nx, ny, nz = (L/discretisation, W/discretisation, 1)
#mesh = from_geofile("film.geo")
mesh = df.Rectangle(-dL, -dW, dL, dW, int(nx), int(ny))
#mesh = df.Box(-dL, -dW, -dH, dL, dW, dH, int(nx), int(ny), int(nz))

def point_contacts(fig=5, debug=False):
    assert fig == 5 or fig == 6

    if fig == 5:
        """
        Two point contacts, on a horizontal line.

        """
        distance = 125.0
        expr = point_contacts_gen([(-distance/2, 0), (distance/2, 0)],
                          radius=point_contact_radius, J=J)
    if fig == 6:
        """
        Three point contacts, arranged on a circle around the center of the mesh.

        """
        distance = 65.0
        r = distance * sqrt(3)/3
        coords = [(r * cos(theta), r * sin(theta)) for theta in [0, 2*pi/3, 4*pi/3]]
        expr = point_contacts_gen(coords, radius=point_contact_radius, J=J)

    if debug:
        df.plot(df.interpolate(expr, sim.S1))
        df.interactive()
        import sys; sys.exit()

    return expr

A = 13.0e-12 # J/m, my guess (Permalloy), paper uses D/(gamma*h_bar), no numerical value

run = 0
for alpha in [0.012, 0.06]: # first one in paper, Section 5 page 8
    for Ms in [640e3, 860e3]: # Section 3 page 4, Section 5 page 8.
        for P in [0.4, 1.0]: # my guessing, no info in paper

            # Should be around 6 nm, Section 3 page 4.
            l_ex = sqrt(2 * A / (mu0 * Ms**2))
            print "The exchange length is l_ex = {:.2} m.".format(l_ex)

            sim = Simulation(mesh, Ms, unit_length=1)
            sim.alpha = alpha
            sim.set_m((0, 0, 1))
            sim.add(Zeeman((0, 0, 1.1 * Ms))) # Section 3, end of Section 5
            sim.add(Exchange(A))
            sim.add(ThinFilmDemag()) # Eq. 2

            # Should be 2 PI * 23 GHz.
            larmor_frequency = 2 * pi * sim.gamma * mu0 * Ms
            print "Larmor frequency is {:.2g}.".format(larmor_frequency)
            time_unit = larmor_frequency / (2 * pi)
            print "1 time unit in the paper is {}s.".format(time_unit)

            # Spin-Torque
            point_contact_radius = 10e-9 # Fig. 5, could be 20e-9 (Sec. 5).
            point_contact_area = pi * point_contact_radius ** 2
            I = 10e-3 # Section 5, page 8.
            J = I / point_contact_area
            print "Current density is J = {:.2} A/m^2.".format(J)
            p = (1, 1, 0)
            d = H
            sim.set_stt(point_contacts(fig=5), P, d, p)
            sim.snapshot(infix=str(run), force_overwrite=True)

            pulse_time = 50e-12
            time_markers = [pulse_time, 65e-12, 225e-12, 315e-12]
            for t in time_markers: 
                print "Running until {}s.".format(t)
                sim.run_until(t)

                if t == pulse_time:
                    print "Switching spin current off at {}s.".format(sim.t)
                    sim.toggle_stt(False)

                print "Saving snapshot of magnetisation at {}s.".format(sim.t)
                sim.snapshot(force_overwrite=True)
            run += 1
