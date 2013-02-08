import dolfin as df
from finmag import Simulation, sim_with
from finmag.energies.zeeman import Zeeman


def macrospin(Ms=0.86e6, m_init=(1, 0, 0), H_ext=(0, 0, 1e6), alpha=0.1,
              name='macrospin'):
    """
    Minimal mesh with two vertices (1 nm apart). No anisotropy,
    exchange coupling or demag is present so that magnetic moments at
    the two vertices behave identical under the influence of the
    external field. (Ideally, we would only have a single vertex but
    Dolfin doesn't support this.)

    Default values for the arguments:

        Ms = 0.86e6  (saturation magnetisation in A/m)

        m_init = (1, 0, 0)  (initial magnetisation pointing along the x-axis)

        H_ext = (0, 0, 1e6)  (external field in A/m)

        alpha = 0.1  (Gilbert damping coefficient)

    """
    mesh = df.UnitInterval()

    sim = Simulation(mesh, Ms=Ms, unit_length=1e-9, name=name)
    sim.alpha = alpha
    sim.set_m(m_init)
    sim.add(Zeeman(H_ext))

    return sim


### XXX The following implementations are only kept for reference.
### They do not work (using sim.run_until() leaves the magnetisation
### unchanged), which seems to indicate that something might be wrong
### with the sim_with() function!? Should investigate this.
###
###    -- Max, 6.2.2013


def macrospin_interval(Ms=0.86e6, m_init=(1, 0, 0), H_ext=(0, 0, 1e6), alpha=0.1, name='macrospin'):
    """
    1d mesh (= interval) with two vertices which are 1 nm apart.

    No anisotropy, exchange coupling or demag is present so that
    magnetic moments at the vertices behave identical under the
    influence of the external field. The damping constant has the
    value alpha=0.1.


    Default values for the arguments:

        Ms = 0.86e6  (saturation magnetisation in A/m)

        m_init = (1, 0, 0)  (initial magnetisation pointing along the x-axis)

        H_ext = (0, 0, 1e6)  (external field in A/m)

        alpha = 0.1  (Gilbert damping coefficient)

    """
    raise NotImplementedError("This implementation doesn't seem to work. Investigate!")
    mesh = df.UnitInterval()
    sim = sim_with(mesh, Ms=1e6, m_init=(1, 0, 0), alpha=alpha,
                   unit_length=1e-9, A=None, demag_solver=None, name=name)
    return sim


def macrospin_box(Ms=0.86e6, m_init=(1, 0, 0), H_ext=(0, 0, 1e6), alpha=0.1, name='macrospin'):
    """
    Cubic mesh of length 1 nm along each edge, with eight vertices
    located in the corners of the cube.

    No anisotropy, exchange coupling or demag is present so that
    magnetic moments at the vertices behave identical under the
    influence of the external field.


    Default values for the arguments:

        Ms = 0.86e6  (saturation magnetisation in A/m)

        m_init = (1, 0, 0)  (initial magnetisation pointing along the x-axis)

        H_ext = (0, 0, 1e6)  (external field in A/m)

        alpha = 0.1  (Gilbert damping coefficient)

    """
    raise NotImplementedError("This implementation doesn't seem to work. Investigate!")
    mesh = df.Box(0, 0, 0, 1, 1, 1, 1, 1, 1)
    sim = sim_with(mesh, Ms=0.86e6, alpha=alpha, unit_length=1e-9,
                   A=None, m_init=(1, 0, 0), name=name)
    return sim
