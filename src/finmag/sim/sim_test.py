import dolfin as df
import numpy as np
import logging
import pytest
import os
import glob
import shutil
from tempfile import mkdtemp
from finmag import sim_with, Simulation
from finmag.example import barmini
from math import sqrt, cos, sin, pi
from finmag.util.helpers import assert_number_of_files
from finmag.sim import sim_helpers
from finmag.energies.zeeman import Zeeman
from finmag.energies.exchange import Exchange

logger = logging.getLogger("finmag")

def num_interactions(sim):
    """
    Helper function to determine the number of interactions present in
    the Simulation.
    """
    return len(sim.llg.effective_field.interactions)

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

    def test_probe_demag_field(self):
        N = self.mesh.num_vertices()
        coords = self.mesh.coordinates()

        # Probe field at all mesh vertices and at the first vertex;
        # also convert a 1d version of the probed vector following
        # dolfin's coordinate convention.
        v_probed = self.sim.probe_field("demag", coords * self.sim.unit_length)
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

    def test_probe_constant_m_at_individual_points(self):
        mesh = df.Box(-2, -2, -2, 2, 2, 2, 5, 5, 5)
        m_init = np.array([0.2, 0.7, -0.4])
        m_init /= np.linalg.norm(m_init)  # normalize the vector for later comparison
        sim = sim_with(mesh, Ms=8.6e5, m_init=m_init, unit_length=1e-9, demag_solver=None)

        # Points inside the mesh where to probe the magnetisation.
        probing_pts = [
            # We are explicitly using integer coordinates for the first
            # point because this used to lead to a very subtle bug.
            [0, 0, 0],
            [0.0, 0.0, 0.0],  # same point again but with float coordinates
            [1e-9, 1e-9, -0.5e-9],
            [-1.3e-9, 0.02e-9, 0.3e-9]]

        # Probe the magnetisation at the given points
        m_probed_vals = [sim.probe_field("m", pt) for pt in probing_pts]

        # Check that we get m_init everywhere.
        for v in m_probed_vals:
            assert(np.allclose(v, m_init))

        # Probe at point outside the mesh
        m_probed_outside = sim.probe_field("m", [5e-9, -6e-9,  1e-9])
        assert(all(np.isnan(m_probed_outside)))

    def test_probe_nonconstant_m_at_individual_points(self):
        TOL=1e-5

        unit_length = 1e-9
        mesh = df.Box(0, 0, 0, 1, 1, 1, 1000, 2, 2)
        m_init = df.Expression(("cos(x[0]*pi)",
                                "sin(x[0]*pi)",
                                "0.0"),
                               unit_length=unit_length)
        sim = sim_with(mesh, Ms=8.6e5, m_init=m_init, unit_length=unit_length, demag_solver=None)

        # Probe the magnetisation along two lines parallel to the x-axis.
        # We choose the limits just inside the mesh boundaries to prevent
        # problems with rounding issues.
        xmin = 0.01 * unit_length
        xmax = 0.99 * unit_length
        y0 = 0.2 * unit_length
        z0 = 0.4 * unit_length
        pts1 = [[x, 0, 0] for x in np.linspace(xmin, xmax, 20)]
        pts2 = [[x, y0, z0] for x in np.linspace(xmin, xmax, 20)]
        probing_pts = np.concatenate([pts1, pts2])

        # Probe the magnetisation at the given points
        m_probed_vals = [sim.probe_field("m", pt) for pt in probing_pts]

        # Check that we get m_init everywhere.
        for i in xrange(len(probing_pts)):
            m = m_probed_vals[i]
            pt = probing_pts[i]
            x = pt[0]
            m_expected = np.array([cos((x / unit_length)*pi),
                                   sin((x / unit_length)*pi),
                                   0.0])
            assert(np.linalg.norm(m - m_expected) < TOL)

    def test_probe_m_on_regular_grid(self):
        """
        Another sanity check using the barmini example; probe the
        magnetisation on a regular 2D grid inside a plane parallel to the
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
        res = sim.probe_field('m', pts * sim.unit_length)

        # Check that 'res' has the right shape and values (the field vectors
        # should be constant and equal to [1/sqrt(2), 0, 1/sqrt(2)].
        assert(res.shape == (nx, ny, 3))
        assert(np.allclose(res[..., 0], 1.0/sqrt(2)))
        assert(np.allclose(res[..., 1], 0.0))
        assert(np.allclose(res[..., 2], 1.0/sqrt(2)))

    def test_schedule(self, tmpdir):
        os.chdir(str(tmpdir))

        # Define a function to be scheduled, with one positional and
        # one keyword argument.
        res = []
        def f(sim, tag, optional=None):
            res.append((tag, int(sim.t * 1e13), optional))

        sim = barmini()
        sim.schedule(f, 'tag1', every=2e-13)
        sim.schedule(f, 'tag2', optional='foo', every=3e-13)
        sim.schedule(f, 'tag3', every=4e-13, optional='bar')
        sim.run_until(1.1e-12)

        assert(sorted(res) ==
               [('tag1', 0, None), ('tag1', 2, None), ('tag1', 4, None),
                ('tag1', 6, None), ('tag1', 8, None), ('tag1', 10, None),
                ('tag2', 0, 'foo'), ('tag2', 3, 'foo'),
                ('tag2', 6, 'foo'), ('tag2', 9, 'foo'),
                ('tag3', 0, 'bar'), ('tag3', 4, 'bar'), ('tag3', 8, 'bar')])

        # The keyword arguments 'at', 'every', 'at_end' and 'realtime'
        # are forbidden:
        def f_illegal_1(sim, at=None): pass
        def f_illegal_2(sim, after=None): pass
        def f_illegal_3(sim, every=None): pass
        def f_illegal_4(sim, at_end=None): pass
        def f_illegal_5(sim, realtime=None): pass
        with pytest.raises(ValueError): sim.schedule(f_illegal_1, at=0.0)
        with pytest.raises(ValueError): sim.schedule(f_illegal_2, at=0.0)
        with pytest.raises(ValueError): sim.schedule(f_illegal_3, at=0.0)
        with pytest.raises(ValueError): sim.schedule(f_illegal_4, at=0.0)
        with pytest.raises(ValueError): sim.schedule(f_illegal_5, at=0.0)


    def test_save_ndt(self, tmpdir):
        os.chdir(str(tmpdir))
        sim = barmini()
        sim.schedule('save_ndt', every=2e-13)
        sim.run_until(1.1e-12)
        a = np.loadtxt('barmini.ndt')
        assert(len(a) == 6)  # we should have saved 6 time steps

    def test_save_restart_data(self, tmpdir):
        """
        Simple test to check that we can save restart data via the
        Simulation class. Note that this basically just checks that we
        can call save_restart_data(). The actual stress-test of the
        functionality is in sim_helpers_test.py.

        """
        os.chdir(str(tmpdir))
        sim = barmini()

        # Test scheduled saving of restart data
        sim.schedule('save_restart_data', at_end=True)
        sim.run_until(1e-13)
        d = sim_helpers.load_restart_data('barmini-restart.npz')
        assert(d['simtime']) == 1e-13

        # Testsaving of restart data using the simulation's own method
        sim.run_until(2e-13)
        sim.save_restart_data()
        d = sim_helpers.load_restart_data('barmini-restart.npz')
        assert(d['simtime']) == 2e-13

    def test_save_vtk(self, tmpdir):
        tmpdir = str(tmpdir)
        sim1_dir = os.path.join(tmpdir, 'sim1')
        sim2_dir = os.path.join(tmpdir, 'sim2')
        sim3_dir = os.path.join(tmpdir, 'sim3')
        sim4_dir = os.path.join(tmpdir, 'sim4')
        os.mkdir(sim1_dir)
        os.mkdir(sim2_dir)
        os.mkdir(sim3_dir)
        os.mkdir(sim4_dir)

        # Run simulation for a few picoseconds and check that the
        # expected vtk files were saved.
        os.chdir(sim1_dir)
        sim = barmini()
        sim.schedule('save_vtk', every=4e-12, at_end=True)
        sim.schedule('save_vtk', at=5e-12)
        sim.run_until(1e-11)
        assert_number_of_files('barmini.pvd', 1)
        assert_number_of_files('barmini*.vtu', 5)

        # Same again, but with a different filename (and a slightly
        # different schedule, just for variety).
        os.chdir(sim2_dir)
        sim = barmini()
        sim.set_vtk_export_filename('m.pvd')
        sim.schedule('save_vtk', every=2e-12)
        sim.schedule('save_vtk', at=3e-12)
        sim.run_until(5e-12)
        assert_number_of_files('m.pvd', 1)
        assert_number_of_files('m*.vtu', 4)
        # Check that no vtk files are present apart from the ones we expect
        assert(len(glob.glob('*.vtu')) + len(glob.glob('*.pvd')) == 4 + 1)

        # Run for a bit, then change the filename and continue
        # running. At the end check that both .pvd files were written
        # correctly.
        os.chdir(sim3_dir)
        sim = barmini()
        sim.set_vtk_export_filename('a.pvd')
        sim.schedule('save_vtk', every=2e-12)
        sim.schedule('save_vtk', at=3e-12)
        sim.run_until(5e-12)
        assert_number_of_files('a.pvd', 1)
        assert_number_of_files('a*.vtu', 4)
        sim.set_vtk_export_filename('b.pvd')
        sim.run_until(10e-12)
        assert_number_of_files('b.pvd', 1)
        assert_number_of_files('b*.vtu', 3)

        sim.save_vtk('c.pvd')
        assert_number_of_files('c.pvd', 1)
        assert_number_of_files('c*.vtu', 1)

    def test_remove_interaction(self):

        mesh = df.Box(0, 0, 0, 1, 1, 1, 1, 1, 1)
        sim = Simulation(mesh, Ms=1, unit_length=1e-9)
        sim.add(Zeeman((0, 0, 1)))
        sim.add(Exchange(13e-12))
        assert(num_interactions(sim) == 2)

        sim.remove_interaction("Exchange")
        assert(num_interactions(sim) == 1)

        sim.remove_interaction("Zeeman")
        assert(num_interactions(sim) == 0)

        # No Zeeman interaction present any more
        with pytest.raises(ValueError):
            sim.remove_interaction("Zeeman")

        # Two different Zeeman interaction present
        sim.add(Zeeman((0, 0, 1)))
        sim.add(Zeeman((0, 0, 2)))
        with pytest.raises(ValueError):
            sim.remove_interaction("Zeeman")

    def test_switch_off_H_ext(self):
        """
        Simply test that we can call sim.switch_off_H_ext()
        """
        mesh = df.Box(0, 0, 0, 1, 1, 1, 1, 1, 1)
        sim = Simulation(mesh, Ms=1, unit_length=1e-9)
        sim.add(Zeeman((1, 2, 3)))

        sim.switch_off_H_ext(remove_interaction=False)
        H = sim.get_interaction("Zeeman").compute_field()
        assert(np.allclose(H, np.zeros_like(H)))

        sim.switch_off_H_ext(remove_interaction=True)
        assert(num_interactions(sim) == 0)
