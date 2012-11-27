import dolfin as df
import numpy as np
import logging
from finmag import sim_with
from finmag.energies import Demag

logger = logging.getLogger("finmag")


def test_get_field_as_dolfin_function():
    """
    Create a simulation, convert the demag field into a dolfin
    function, evaluate it a all nodes and check that the resulting
    vector of the field values is the same as the one internally
    stored in the simulation.
    """
    mesh = df.Box(0, 0, 0, 1, 1, 1, 5, 5, 5)
    sim = sim_with(mesh, Ms=8.6e5, m_init=(1, 0, 0), alpha=1.0,
                   unit_length=1e-9, A=13.0e-12, demag_solver='FK')
    sim.relax()

    fun_demag = sim.get_field_as_dolfin_function("demag")

    # Evalute the field function at all mesh vertices. This gives a
    # Nx3 array, which we convert back into a 1D array using dolfin's
    # convention of arranging coordinates.
    v_eval = np.array([fun_demag(c) for c in mesh.coordinates()])
    v_eval_1d = np.concatenate([v_eval[:, 0], v_eval[:, 1], v_eval[:, 2]])

    # Now check that this is essentially the same as vector of the
    # original demag interaction.
    demag = sim.get_interaction("demag")
    v_demag = demag.compute_field()

    assert(np.allclose(v_demag, v_eval_1d))

    # Note that we cannot use '==' for the comparison above because
    # the function evaluation introduced numerical inaccuracies:
    logger.debug("Are the vectors identical? "
                 "{}".format((v_demag == v_eval_1d).all()))


if __name__ == '__main__':
    test_get_field_as_dolfin_function()
