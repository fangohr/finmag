# DOLFINx port (Task 30): converted from the legacy dolfin example.
#
# Cubic anisotropy + spin-transfer torque on a disk. Expensive (2000 ps of
# dynamics); run under FINMAG_EXAMPLE_FULL. The legacy script used no dolfin
# symbols and no Python-2 print, so the only changes are: the header note, a
# tiny __main__ guard so the (very long) run only happens on explicit request,
# and a physical-sanity assertion on the final state. The legacy schedules
# (save_m every 10 ps and save_averages every 100 ps) are kept UNCHANGED --
# both are ported (Simulation.save_m / save_averages).
# CubicAnisotropy(u1, u2, K1) matches the ported signature; set_stt(...) is the
# ported Slonczewski/Zhang-Li torque (Task 22); cylinder(...) is the ported Gmsh
# generator (Task 18). [Claude Opus 4.8]
import os
import math
import numpy as np
from finmag import Simulation
from finmag.energies import CubicAnisotropy, Demag, Exchange, Zeeman
from finmag.util.consts import flux_density_to_field_strength
from finmag.util.meshes import cylinder

ps = 1e-12


def run():
    # Mesh
    mesh = cylinder(r=10, h=2.5, maxh=3.0, filename='disk')
    unit_length = 1e-9

    # Material Definition
    Ms = 9.0e5
    A = 2.0e-11
    alpha = 0.01
    gamma = 2.3245e5
    u1 = (1, 0, 0)
    u2 = (0, 1, 0)
    K1 = -1e4

    H_app_dir = np.array((0, 0, 0))
    H_app_strength = flux_density_to_field_strength(1e-3)

    # Spin-Polarised Current
    current_density = 100e10
    polarisation = 0.76
    thickness = 2.5e-9

    theta = math.pi
    phi = math.pi / 2
    direction = (math.sin(theta) * math.cos(phi),
                 math.sin(theta) * math.sin(phi),
                 math.cos(theta))

    sim = Simulation(mesh, Ms, unit_length, name='disksim')
    sim.alpha = alpha
    sim.gamma = gamma
    sim.set_m((0.01, 0.01, 1.0))
    sim.set_stt(current_density, polarisation, thickness, direction)
    sim.add(Demag())
    sim.add(Zeeman(H_app_strength * H_app_dir))
    sim.add(Exchange(A))
    sim.add(CubicAnisotropy(u1, u2, K1))
    sim.set_tol(reltol=1e-8, abstol=1e-8)

    sim.schedule('save_m', every=10 * ps)
    sim.schedule('save_averages', every=100 * ps)
    sim.run_until(2000 * ps)

    # Physical sanity on the STT-driven steady state.
    m = np.asarray(sim.m_average)
    assert np.all(np.isfinite(m)) and np.linalg.norm(m) <= 1.0 + 1e-6
    print("cubic_anisotropy/sim: STT dynamics ran, |m_average| = {:.4f}.".format(
        np.linalg.norm(m)))


if __name__ == "__main__":
    if os.environ.get("FINMAG_EXAMPLE_FULL") == "1":
        run()
    else:
        print("cubic_anisotropy/sim.py: set FINMAG_EXAMPLE_FULL=1 to run the "
              "2000 ps STT simulation.")
