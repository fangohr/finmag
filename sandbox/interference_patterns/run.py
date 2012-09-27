import dolfin as df
import numpy as np
from math import pi, sin, cos, sqrt
from finmag import Simulation
from finmag.util.meshes import from_geofile
from finmag.energies import Zeeman, Exchange, ThinFilmDemag
from point_contacts import point_contacts as point_contacts_gen

def point_contacts(fig=5, debug=False):
    assert fig == 5 or fig == 6
    if fig == 5: # Two point contacts, on a horizontal line.
        distance = 125.0
        expr = point_contacts_gen([(-distance/2, 0), (distance/2, 0)], radius=pc_radius, J=J/2)
    if fig == 6: # Three point contacts, arranged on a circle around the center of the mesh.
        distance = 65.0
        r = distance * sqrt(3)/3
        coords = [(r * cos(theta), r * sin(theta)) for theta in [0, 2*pi/3, 4*pi/3]]
        expr = point_contacts_gen(coords, radius=pc_radius, J=J/3)
    if debug:
        df.plot(df.interpolate(expr, sim.S1))
        df.interactive()
        import sys; sys.exit()
    return expr

unit_length = 1e-9;
Ms = 860e3
A = 17e-12 # J/m, computed from the exchange length, is meant to be Permalloy

# Point contacts
pc_radius = 10
pc_area = pi * (pc_radius * unit_length) ** 2
I = 10e-3 # Section 5, page 8.
J = I / pc_area
print "Current density is J = {:.2} A/m^2.".format(J)

for figure in [5, 6]:
    if figure == 5:
        mesh = from_geofile("film.geo")
    else:
        mesh = df.Rectangle(-220, -220, 220, 220, 400, 400)

    sim = Simulation(mesh, Ms, unit_length)
    sim.alpha = 0.012 # Section 5 page 8
    sim.set_m((0, 0, 1))
    sim.add(Zeeman((0, 0, 1.1 * Ms))) # Section 3, end of Section 5
    sim.add(Exchange(A))
    sim.add(ThinFilmDemag())
    sim.set_stt(point_contacts(figure), polarisation=1.0, thickness=5e-9, direction=(1,1,0))
    
    m = df.Function(sim.S3)
    m_archive = df.File("figure_{}/m.pvd".format(figure))

    t = 0; dt = 5e-12; pulse_time = 10e-12; t_max = 100e-12;
    while t <= t_max:
        print "Running until {}s.".format(t)
        sim.run_until(t)

        if abs(t - pulse_time) < 1e-15:
            print "Switching spin current off at {}s.".format(sim.t)
            sim.toggle_stt(False)
    
        m.vector()[:] = np.sqrt(1 - sim.m**2)
        m_archive << m
        t += dt;
    del(sim)
