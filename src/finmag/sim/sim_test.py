import dolfin as df
import numpy as np
import logging
import os
import glob
import shutil
from tempfile import mkdtemp
from finmag import sim_with
from finmag.example import barmini
from math import sqrt

logger = logging.getLogger("finmag")

class TestSimulation(object):
    @classmethod
    def setup_class(cls):
        # N.B.: The mesh and simulation are only created once for the
        # entire test class and are re-used in each test method for
        # efficiency. Thus they should be regarded as read-only and
        # not be changed in any test method, otherwise there may be
        # unpredicted bugs or errors in unrelated test methods!
        cls.mesh = df.BoxMesh(0, 0, 0, 1, 1, 1, 5, 5, 5)
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
        print "v0_probed: {}".format(v0_probed)
        print "v0_ref: {}".format(v0_ref)
        assert(np.allclose(v0_probed, v0_ref))
        assert(np.allclose(v_probed_1d, v_ref))

    def test_probe_field2(self):
        """
        Another sanity check using the barmini example; probe the
        field on a regular 2D grid inside a plane parallel to the
        x/y-plane (with different numbers of probing points in x- and
        y-direction).
        """

        # Set up the simulation
        sim = barmini()
        nx = 5
        ny = 10
        z = 5.0  # use cutting plane in the middle of the cuboid
        X, Y = np.mgrid[0:3:nx*1j, 0:3:ny*1j]
        pts = np.array([[(X[i, j], Y[i,j], z) for j in xrange(ny)] for i in xrange(nx)])

        # Probe the field
        res = sim.probe_field('m', pts)

        # Check that 'res' has the right shape and values (the field vectors
        # should be constant and equal to [1/sqrt(2), 0, 1/sqrt(2)].
        assert(res.shape == (nx, ny, 3))
        assert(np.allclose(res[..., 0], 1.0/sqrt(2)))
        assert(np.allclose(res[..., 1], 0.0))
        assert(np.allclose(res[..., 2], 1.0/sqrt(2)))


def test_relax_with_saving_snapshots():
    mesh = df.BoxMesh(0, 0, 0, 1, 1, 1, 5, 5, 5)
    tmpdir = mkdtemp()
    sim = sim_with(mesh, Ms=8.6e5, m_init=(1, 0, 0), alpha=1.0,
                   unit_length=1e-9, A=13.0e-12, demag_solver='FK')
    sim.relax(save_snapshots=True, filename=os.path.join(tmpdir, 'sim.pvd'),
              save_every=5e-14, save_final_snapshot=True, force_overwrite=True)

    # Check that the relaxation took as long as expected
    assert(np.allclose(sim.t, 2.078125e-13, atol=0))

    # Checkt that the correct number of snapshots was saved: 5 due to
    # 'save_every=5e-14' and 1 due to 'save_final_snapshots=True'.
    vtu_files = glob.glob(os.path.join(tmpdir, "*.vtu"))
    assert(len(vtu_files) == 5 + 1)

    shutil.rmtree(tmpdir)
