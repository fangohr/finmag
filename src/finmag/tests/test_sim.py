import dolfin as df
import numpy as np
import logging
from finmag import sim_with
from finmag.energies import Demag

logger = logging.getLogger("finmag")

class TestSimulation(object):
    @classmethod
    def setup_class(cls):
        # N.B.: The mesh and simulation are only created once for the
        # entire test class and are re-used in each test method for
        # efficiency. Thus they should be regarded as read-only and
        # not be changed in any test method, otherwise there may be
        # unpredicted bugs or errors in unrelated test methods!
        cls.mesh = df.Box(0, 0, 0, 1, 1, 1, 5, 5, 5)
        cls.sim = sim_with(cls.mesh, Ms=8.6e5, m_init=(1, 0, 0), alpha=1.0,
                           unit_length=1e-9, A=13.0e-12, demag_solver='FK')
        cls.sim.relax()

    def test_get_field_as_dolfin_function(self):
        """
        Convert the demag field into a dolfin function, evaluate it a all
        nodes and check that the resulting vector of the field values is
        the same as the one internally stored in the simulation.
        """
        fun_demag = self.sim.get_field_as_dolfin_function("demag")

        # Evalute the field function at all mesh vertices. This gives a
        # Nx3 array, which we convert back into a 1D array using dolfin's
        # convention of arranging coordinates.
        v_eval = np.array([fun_demag(c) for c in self.mesh.coordinates()])
        v_eval_1d = np.concatenate([v_eval[:, 0], v_eval[:, 1], v_eval[:, 2]])

        # Now check that this is essentially the same as vector of the
        # original demag interaction.
        demag = self.sim.get_interaction("demag")
        v_demag = demag.compute_field()

        assert(np.allclose(v_demag, v_eval_1d))

        # Note that we cannot use '==' for the comparison above because
        # the function evaluation introduced numerical inaccuracies:
        logger.debug("Are the vectors identical? "
                     "{}".format((v_demag == v_eval_1d).all()))

    def test_probe_field(self):
        N = self.mesh.num_vertices()
        coords = self.mesh.coordinates()

        # Probe field at all mesh vertices and at the first vertex;
        # also convert a 1d version of the probed vector following
        # dolfin's coordinate convention.
        v_probed = self.sim.probe_field("demag", coords)
        v_probed_1d = np.concatenate([v_probed[:, 0],
                                      v_probed[:, 1],
                                      v_probed[:, 2]])
        v0_probed = self.sim.probe_field("demag", coords[0])

        # Create reference vectors at the same positions
        v_ref = self.sim.get_interaction("demag").compute_field()
        v0_ref = v_ref[[0, N, 2*N]]

        # Check that the results coincide
        assert(np.allclose(v0_probed, v0_ref))
        assert(np.allclose(v_probed_1d, v_ref))
