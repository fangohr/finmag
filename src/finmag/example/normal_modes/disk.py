import finmag
import dolfin as df
import os
from finmag.util.mesh_templates import Nanodisk


def disk(d=60, h=10, maxh=5.0, relaxed=True, name='normal_modes_nanodisk', A=13e-12, H_ext_relax=[1e5, 1e3, 0], H_ext_ringdown=[1e5, 0, 0],
         demag_solver='FK', demag_solver_type=None, force_relaxation=False):
    """
    Permalloy nanodisk with diameter d=60 nm, height h=10 nm and mesh
    discretisation maxh=5.0. An external field of strength 100 kA/m is
    applied along the x-axis, with a small (1 kA/m) y-component for
    the relaxation which is then removed for the ringdown phase.

    """
    # I don't know it's suitable to move the global definition to local one
    # just try to make the cython happy?
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    if d == 60 and h == 10 and maxh == 5.0:
        # Use precomputed standard mesh
        mesh = df.Mesh(
            os.path.join(MODULE_DIR, 'disk__d_60__h_10__maxh_5.xml.gz'))
    else:
        mesh = Nanodisk(d=d, h=h).create_mesh(maxh=maxh)

    # Material parameters for Permalloy
    Ms = 8e5
    m_init = [1, 0, 0]
    alpha_relax = 1.0
    sim = finmag.normal_mode_simulation(mesh, Ms, m_init, alpha=alpha_relax, unit_length=1e-9, A=A, H_ext=H_ext_relax,
                                        demag_solver=demag_solver, demag_solver_type=demag_solver_type, name=name)

    if relaxed:
        if not force_relaxation and (d == 60 and h == 10 and maxh == 5.0):
            sim.restart(os.path.join(MODULE_DIR, 'disk_relaxed.npz'))
        else:
            sim.relax()

    alpha_ringdown = 0.01
    t_end = 1e-9
    save_ndt_every = 1e-11
    save_m_every = 1e-11
    m_snapshots_filename = 'snapshots/m_ringdown.npy'

    # Pre-define parameters for the 'run_ringdown()' method so that
    # the user can just say: 'sim.run_ringdown()' and it does
    # something sensible.
    def ringdown(t_end=1e-9, alpha=alpha_ringdown, H_ext=H_ext_ringdown,
                 reset_time=True, clear_schedule=True,
                 save_ndt_every=save_ndt_every, save_vtk_every=None, save_m_every=save_m_every,
                 vtk_snapshots_filename=None, m_snapshots_filename=m_snapshots_filename,
                 overwrite=False):
        sim.run_ringdown(
            t_end=t_end, alpha=alpha, H_ext=H_ext, reset_time=reset_time,
            clear_schedule=clear_schedule, save_ndt_every=save_ndt_every,
            save_vtk_every=None, save_m_every=save_m_every,
            vtk_snapshots_filename=None, m_snapshots_filename=m_snapshots_filename,
            overwrite=overwrite)
    sim.ringdown = ringdown

    return sim
