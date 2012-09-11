import dolfin as df
import numpy as np
from math import pi, sin, cos, sqrt
from finmag import Simulation
from finmag.util.consts import mu0, h_bar
from finmag.util.meshes import from_geofile
from finmag.energies import Zeeman, Exchange, ThinFilmDemag
from point_contacts import point_contacts as point_contacts_gen

def point_contacts(fig=5, debug=False):
    assert fig == 5 or fig == 6
    if fig == 5: # Two point contacts, on a horizontal line.
        distance = 125.0
        expr = point_contacts_gen([(-distance/2, 0), (distance/2, 0)], radius=point_contact_radius, J=J)
    if fig == 6: # Three point contacts, arranged on a circle around the center of the mesh.
        distance = 65.0
        r = distance * sqrt(3)/3
        coords = [(r * cos(theta), r * sin(theta)) for theta in [0, 2*pi/3, 4*pi/3]]
        expr = point_contacts_gen(coords, radius=point_contact_radius, J=J)
    if debug:
        df.plot(df.interpolate(expr, sim.S1))
        df.interactive()
        import sys; sys.exit()
    return expr

#mesh = df.Rectangle(-220, -220, 220, 220, 100, 100)
mesh = from_geofile("film.geo")
unit_length = 1e-9
Ms = 860e3 # 640e3 Section 3 page 4, 860e3 Section 5 page 8.

for figure in [5, 6]:

    sim = Simulation(mesh, Ms, unit_length)
    sim.alpha = 0.012 # Section 5 page 8
    sim.set_m((0, 0, 1))
    sim.add(Zeeman((0, 0, 1.1 * Ms))) # Section 3, end of Section 5
    A = 13.0e-12 # J/m, my guess (Permalloy), paper uses D/(gamma*h_bar), no numerical value
    sim.add(Exchange(A))
    sim.add(ThinFilmDemag()) # Eq. 2

    # Spin-Torque
    point_contact_radius = 10 # Fig. 5, could be 20 (Sec. 5).
    point_contact_area = pi * (point_contact_radius * unit_length) ** 2
    I = 10e-3 # Section 5, page 8.
    J = I / point_contact_area
    print "Current density is J = {:.2} A/m^2.".format(J)
    sim.set_stt(point_contacts(figure), polarisation=0.4, thickness=5e-9, direction=(1, 0, 0))
    
    m_abs = df.Function(sim.S3)

    m_archive = df.File("figure_{}/m.pvd".format(figure))
    t = 0; dt = 5e-12; pulse_time = 50e-12; t_max = 200e-12;
    while t <= t_max:
        print "Running until {}s.".format(t)
        sim.run_until(t)

        if abs(t - pulse_time) < 1e-15:
            print "Switching spin current off at {}s.".format(sim.t)
            sim.toggle_stt(False)

        m_abs.vector()[:] = np.abs(sim.m)
        m_archive << m_abs 
        t += dt;
    del(sim)
