import dolfin as df
import numpy as np
import logging
import pytest
import os
from glob import glob
from distutils.version import StrictVersion
from finmag import sim_with, Simulation, set_logging_level, normal_mode_simulation
from finmag.example import barmini
from math import sqrt, cos, sin, pi
from finmag.util.helpers import assert_number_of_files, vector_valued_function, logging_status_str
from finmag.util.meshes import nanodisk
from finmag.sim import sim_helpers
from finmag.energies import Zeeman, TimeZeeman, Exchange, UniaxialAnisotropy
from finmag.util.fileio import Tablereader
from finmag.util.timings import default_timer
from finmag.util.ansistrm import ColorizingStreamHandler

logger = logging.getLogger("finmag")
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


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
        cls.mesh = df.BoxMesh(0, 0, 0, 1, 1, 1, 5, 5, 5)
        cls.sim = sim_with(cls.mesh, Ms=8.6e5, m_init=(1, 0, 0), alpha=1.0,
                           unit_length=1e-9, A=13.0e-12, demag_solver='FK')
        cls.sim.relax()

    def test_get_interaction(self):
        sim = sim_with(self.mesh, Ms=8.6e5, m_init=(1, 0, 0), alpha=1.0,
                       unit_length=1e-9, A=13.0e-12, demag_solver='FK')

        # These should just work
        sim.get_interaction('Exchange')
        sim.get_interaction('Demag')

        with pytest.raises(ValueError):
            sim.get_interaction('foobar')

        exch = Exchange(A=13e-12, name='foobar')
        sim.add(exch)
        assert exch == sim.get_interaction('foobar')

    def test_compute_energy(self):
        sim = sim_with(self.mesh, Ms=8.6e5, m_init=(1, 0, 0), alpha=1.0,
                       unit_length=1e-9, A=13.0e-12, demag_solver='FK')

        # These should just work
        sim.compute_energy('Exchange')
        sim.compute_energy('Demag')
        sim.compute_energy('Total')
        sim.compute_energy('total')

        # A non-existing interaction should return zero energy
        assert(sim.compute_energy('foobar') == 0.0)

        new_exch = Exchange(A=13e-12, name='foo')
        sim.add(new_exch)
        assert new_exch.compute_energy() == sim.compute_energy('foo')

    def test_get_field_as_dolfin_function(self):
        """
        Convert the demag field into a dolfin function, evaluate it a all
        nodes and check that the resulting vector of the field values is
        the same as the one internally stored in the simulation.
        """
        fun_demag = self.sim.get_field_as_dolfin_function("Demag")

        # Evalute the field function at all mesh vertices. This gives a
        # Nx3 array, which we convert back into a 1D array using dolfin's
        # convention of arranging coordinates.
        v_eval = np.array([fun_demag(c) for c in self.mesh.coordinates()])
        v_eval_1d = np.concatenate([v_eval[:, 0], v_eval[:, 1], v_eval[:, 2]])

        # Now check that this is essentially the same as vector of the
        # original demag interaction.
        demag = self.sim.get_interaction("Demag")
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
        v_probed = self.sim.probe_field("Demag", coords * self.sim.unit_length)
        v_probed_1d = np.concatenate([v_probed[:, 0],
                                      v_probed[:, 1],
                                      v_probed[:, 2]])
        v0_probed = self.sim.probe_field("Demag", coords[0])

        # Create reference vectors at the same positions
        v_ref = self.sim.get_interaction("Demag").compute_field()
        v0_ref = v_ref[[0, N, 2*N]]

        # Check that the results coincide
        print "v0_probed: {}".format(v0_probed)
        print "v0_ref: {}".format(v0_ref)
        assert(np.allclose(v0_probed, v0_ref))
        assert(np.allclose(v_probed_1d, v_ref))

    def test_probe_constant_m_at_individual_points(self):
        mesh = df.BoxMesh(-2, -2, -2, 2, 2, 2, 5, 5, 5)
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
        assert((np.ma.getmask(m_probed_outside) == True).all())

    def test_probe_nonconstant_m_at_individual_points(self):
        TOL=1e-5

        unit_length = 1e-9
        mesh = df.BoxMesh(0, 0, 0, 1, 1, 1, 1000, 2, 2)
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
        f = Tablereader('barmini.ndt')
        assert(len(f.timesteps()) == 6)  # we should have saved 6 time steps

        # Also assert that the energy and field terms are written automatically
        entities = f.entities()
        assert 'E_Exchange' in entities
        assert 'H_Exchange_x' in entities
        assert 'H_Exchange_y' in entities
        assert 'H_Exchange_z' in entities
        assert 'E_Demag' in entities
        assert 'H_Demag_x' in entities
        assert 'H_Demag_y' in entities
        assert 'H_Demag_z' in entities


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

    def test_restart(self, tmpdir):
        os.chdir(str(tmpdir))

        # Run the simulation for 1 ps; save restart data; reload it
        # while resetting the simulation time to 0.5 ps; then run
        # again for 1 ps until we have reached 1.5 ps.
        sim1 = barmini(name='barmini1')
        sim1.schedule('save_ndt', every=1e-13)
        sim1.run_until(1e-12)
        sim1.save_restart_data('barmini.npz')
        sim1.restart('barmini.npz', t0=0.5e-12)
        sim1.run_until(1.5e-12)

        # Check that the time steps for sim1 are as expected
        f1 = Tablereader('barmini1.ndt')
        t1 = f1.timesteps()
        t1_expected = np.concatenate([np.linspace(0, 1e-12, 11),
                                      np.linspace(0.5e-12, 1.5e-12, 11)])
        assert(np.allclose(t1, t1_expected, atol=0))

        # Run a second simulation for 2 ps continuously
        sim2 = barmini(name='barmini2')
        sim2.schedule('save_ndt', every=1e-13)
        sim2.run_until(2e-12)

        # Check that the time steps for sim1 are as expected
        f2 = Tablereader('barmini2.ndt')
        t2 = f2.timesteps()
        t2_expected = np.linspace(0, 2e-12, 21)
        assert(np.allclose(t2, t2_expected, atol=0))

        # Check that the magnetisation dynamics of sim1 and sim2 are
        # the same and that we end up with the same magnetisation.
        for col in ['m_x', 'm_y', 'm_z']:
            a = f1[col]
            b = f2[col]
            a = np.concatenate([a[:10], a[11:]])  # delete the duplicate line due to the restart
            assert(np.allclose(a, b))
        assert(np.allclose(sim1.m, sim2.m, atol=1e-6))

        # Check that resetting the time to zero works (this used to not work due to a bug).
        sim1.restart('barmini.npz', t0=0.0)
        assert(sim1.t == 0.0)

    def test_reset_time(self, tmpdir):
        """
        Integrate a simple macrospin simulation for a bit, then reset
        the time to an earlier point in time and integrate again. We
        expect the same magnetisation dynamics as if the simulation
        had run for the same length continuously, but the internal
        clock times should be different.

        """
        os.chdir(str(tmpdir))

        # First simulation: run for 50 ps, reset the time to 30 ps and run
        # again for 50 ps.
        mesh = df.BoxMesh(0, 0, 0, 1, 1, 1, 1, 1, 1)
        sim1 = Simulation(mesh, Ms=1, name='test_save_ndt')
        sim1.alpha = 0.05
        sim1.set_m((1, 0, 0))
        sim1.add(Zeeman((0, 0, 1e6)))

        sim1.schedule('save_ndt', every=1e-11)
        sim1.run_until(5e-11)
        sim1.reset_time(3e-11)
        sim1.run_until(8e-11)

        # Run a second simulation for 100 ps continuously, without
        # resetting the time in between.
        sim2 = Simulation(mesh, Ms=1, name='test_save_ndt2')
        sim2.alpha = 0.05
        sim2.set_m((1, 0, 0))
        sim2.add(Zeeman((0, 0, 1e6)))

        sim2.schedule('save_ndt', every=1e-11)
        sim2.run_until(10e-11)

        # Check that the time steps for sim1 are as expected.
        a = np.loadtxt('test_save_ndt.ndt')
        ta = a[:, 0]
        ta_expected = 1e-11 * np.array([0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8])
        assert(np.allclose(ta, ta_expected, atol=0))

        # Check that the time steps for sim2 are as expected.
        b = np.loadtxt('test_save_ndt2.ndt')
        tb = b[:, 0]
        tb_expected = 1e-11 * np.arange(10+1)
        assert(np.allclose(tb, tb_expected, atol=0))

        # Delete the duplicate line due to resetting the time
        a = np.concatenate([a[:5, :], a[6:, :]])

        # Check that the magnetisation dynamics of sim1 and sim2 are the same.
        assert(np.allclose(a[:, 1:], b[:, 1:], atol=1e-4, rtol=1e-8))

    def test_save_vtk(self, tmpdir):
        os.chdir(str(tmpdir))

        # Create an empty dummy .pvd file
        with open('barmini.pvd', 'w'): pass

        sim = barmini()
        with pytest.raises(IOError):
            # existing file should not be overwritten
            sim.schedule('save_vtk', every=1e-13, overwrite=False)
            #sim.run_until(1e-12)

        # This time we enforce overwriting
        sim = barmini()
        sim.schedule('save_vtk', every=1e-13, overwrite=True)
        sim.run_until(5e-13)
        assert(len(glob('barmini*.vtu')) == 6)

        # Schedule saving to different filenames and at various time steps.
        sim = barmini()
        sim.schedule('save_vtk', filename='bar2.pvd', every=1e-13, overwrite=True)
        sim.schedule('save_vtk', filename='bar2.pvd', at=2.5e-13, overwrite=False)
        sim.schedule('save_vtk', filename='bar2.pvd', at=4.3e-13, overwrite=False)

        sim.schedule('save_vtk', filename='bar3.pvd', at=3.5e-13, overwrite=True)
        sim.schedule('save_vtk', filename='bar3.pvd', every=2e-13, overwrite=True)
        sim.schedule('save_vtk', filename='bar3.pvd', at=3.8e-13, overwrite=True)

        # ... then run the simulation
        sim.run_until(5e-13)

        # ... also save another vtk snapshot manually
        sim.save_vtk(filename='bar3.pvd')

        # ... and check that the correct number of files was created
        assert(len(glob('bar2*.vtu')) == 8)
        assert(len(glob('bar3*.vtu')) == 6)

        # Saving another snapshot with overwrite=True should erase existing .vtu files
        sim.save_vtk(filename='bar3.pvd', overwrite=True)
        assert(len(glob('bar3*.vtu')) == 1)

    def test_sim_schedule_clear(self, tmpdir):
        os.chdir(str(tmpdir))

        # Run simulation for a few picoseconds and check that the
        # expected vtk files were saved.
        sim = barmini()
        sim.schedule('save_vtk', every=2e-12)
        sim.schedule('save_vtk', at=3e-12)
        sim.run_until(5e-12)
        assert_number_of_files('barmini.pvd', 1)
        assert_number_of_files('barmini*.vtu', 4)

        # Clear schedule and continue running; assert that no
        # additional files were saved
        sim.clear_schedule()
        sim.run_until(10e-12)
        assert_number_of_files('barmini.pvd', 1)
        assert_number_of_files('barmini*.vtu', 4)

        # Schedule saving to a different filename and run further
        sim.schedule('save_vtk', filename='a.pvd', every=3e-12)
        sim.schedule('save_vtk', filename='a.pvd', at=14e-12)
        sim.run_until(20e-12)
        assert_number_of_files('a.pvd', 1)
        assert_number_of_files('a*.vtu', 5)

    def test_remove_interaction(self):

        mesh = df.BoxMesh(0, 0, 0, 1, 1, 1, 1, 1, 1)
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

        # Two different Zeeman interactions present
        sim.add(Zeeman((0, 0, 1)))
        sim.add(Zeeman((0, 0, 2), name="Zeeman2"))
        sim.remove_interaction("Zeeman")
        sim.remove_interaction("Zeeman2")

    def test_switch_off_H_ext(self):
        """
        Simply test that we can call sim.switch_off_H_ext()
        """
        mesh = df.BoxMesh(0, 0, 0, 1, 1, 1, 1, 1, 1)
        sim = Simulation(mesh, Ms=1, unit_length=1e-9)
        sim.add(Zeeman((1, 2, 3)))

        sim.switch_off_H_ext(remove_interaction=False)
        H = sim.get_interaction("Zeeman").compute_field()
        assert(np.allclose(H, np.zeros_like(H)))

        sim.switch_off_H_ext(remove_interaction=True)
        assert(num_interactions(sim) == 0)

    def test_set_H_ext(self):
        mesh = df.BoxMesh(0, 0, 0, 1, 1, 1, 1, 1, 1)
        sim = Simulation(mesh, Ms=1, unit_length=1e-9)
        sim.add(Zeeman((1, 2, 3)))

        H = sim.get_field_as_dolfin_function('Zeeman').vector().array()
        H = sim.probe_field('Zeeman', [0.5e-9, 0.5e-9, 0.5e-9])
        assert(np.allclose(H, [1, 2, 3]))

        sim.set_H_ext([-4, -5, -6])
        H = sim.probe_field('Zeeman', [0.5e-9, 0.5e-9, 0.5e-9])
        assert(np.allclose(H, [-4, -5, -6]))

        # Try to set H_ext in a simulation that doesn't have a Zeeman interaction yet
        sim = Simulation(mesh, Ms=1, unit_length=1e-9)
        sim.set_H_ext((1, 2, 3))  # this should not raise an error!
        H = sim.get_field_as_dolfin_function('Zeeman').vector().array()
        H = sim.probe_field('Zeeman', [0.5e-9, 0.5e-9, 0.5e-9])
        assert(np.allclose(H, [1, 2, 3]))

    @pytest.mark.skipif("not StrictVersion(df.__version__) < StrictVersion('1.2.0')")
    def test_pbc2d_m_init(self):

        def m_init_fun(pos):
            if pos[0]==0 or pos[1]==0:
                return [0,0,1]
            else:
                return [0,0,-1]

        mesh = df.UnitSquareMesh(3, 3)

        m_init = vector_valued_function(m_init_fun, mesh)
        sim = Simulation(mesh, Ms=1, pbc2d=True)
        sim.set_m(m_init)
        expect_m=np.zeros((3,16))
        expect_m[2,:]=np.array([1, 1, 1, 1, 1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1,  1])
        expect_m.shape=(48,)

        assert np.array_equal(sim.m,expect_m)

    def test_set_stt(self):
        """
        Simple macrospin simulation with STT where the current density
        changes sign halfway through the simulation.
        """
        import matplotlib.pyplot as plt
        mesh = df.BoxMesh(0, 0, 0, 1, 1, 1, 1, 1, 1)
        sim = Simulation(mesh, Ms=8.6e5, unit_length=1e-9, name='macrospin_with_stt')
        sim.m = (1, 0, 0)
        sim.add(Zeeman([0, 0, 1e5]))
        sim.alpha = 0.0  # no damping

        def J(t):
            return 0.5e11 if (t < 2.5e-9) else -0.5e11

        sim.set_stt(0.05e11, 1.0, 2e-9, (0, 0, 1), with_time_update=J)
        sim.schedule('save_ndt', every=1e-11)
        sim.run_until(5e-9)

        ts, xs, ys, zs = np.loadtxt('macrospin_with_stt.ndt').T[:4]
        fig = plt.figure(figsize=(20, 5))
        ax1 = fig.add_subplot(131); ax1.plot(ts, xs)
        ax2 = fig.add_subplot(132); ax2.plot(ts, ys)
        ax3 = fig.add_subplot(133); ax3.plot(ts, zs)
        fig.savefig('macrospin_with_stt.png')

        # Assert that the dynamics of m_z are symmetric over time. In
        # theory, this should also be true of m_x and m_y, but since
        # they oscillate rapidly there is quite a bit of numerical
        # inaccuracy, so we're only testing for m_z here.
        assert max(abs(zs - zs[::-1])) < 0.005

    def test_mesh_info(self):
        mesh = df.BoxMesh(0, 0, 0, 1, 1, 1, 1, 1, 1)
        Ms = 8.6e5
        unit_length = 1e-9

        # Simulation without exchange/anisotropy
        sim1 = Simulation(mesh, Ms, unit_length)
        print sim1.mesh_info()

        # Simulation with exchange but without anisotropy
        sim2 = Simulation(mesh, Ms, unit_length)
        sim2.add(Exchange(A=13e-12))
        print sim2.mesh_info()

        # Simulation with anisotropy but without exchange
        sim3 = Simulation(mesh, Ms, unit_length)
        sim3.add(UniaxialAnisotropy(K1=520e3, axis=[0, 0, 1]))
        print sim3.mesh_info()

        # Simulation with both exchange and anisotropy
        sim4 = Simulation(mesh, Ms, unit_length)
        sim4.add(Exchange(A=13e-12))
        sim4.add(UniaxialAnisotropy(K1=520e3, axis=[0, 0, 1]))
        pytest.xfail("print sim4.mesh_info()")


    def test_save_field(self, tmpdir):
        os.chdir(str(tmpdir))
        sim = barmini()

        # Save the magnetisation using the default filename
        sim.save_field('m')
        sim.save_field('m')
        sim.save_field('m')
        assert(len(glob('barmini_m*.npy')) == 1)
        os.remove('barmini_m.npy')

        # Save incrementally
        sim.save_field('m', incremental=True)
        sim.save_field('m', incremental=True)
        sim.save_field('m', incremental=True)
        assert(len(glob('barmini_m_[0-9]*.npy')) == 3)

        # Check that the 'overwrite' keyword works
        sim2 = barmini()
        with pytest.raises(IOError):
            sim2.save_field('m', incremental=True)
        sim2.save_field('m', incremental=True, overwrite=True)
        sim2.save_field('m', incremental=True)
        assert(len(glob('barmini_m_[0-9]*.npy')) == 2)

        sim.save_field('Demag', incremental=True)
        assert(os.path.exists('barmini_demag_000000.npy'))
        sim.save_field('Demag')
        assert(os.path.exists('barmini_demag.npy'))

        sim.save_field('Demag', filename='demag.npy', incremental=True)
        assert(os.path.exists('demag_000000.npy'))

    def test_save_m(self, tmpdir):
        """
        Similar test as 'test_save_field', but for the convenience shortcut 'save_m'.
        """
        os.chdir(str(tmpdir))
        sim = barmini()

        # Save the magnetisation using the default filename
        sim.save_m()
        sim.save_m()
        sim.save_m()
        assert(len(glob('barmini_m*.npy')) == 1)
        os.remove('barmini_m.npy')

        # Save incrementally
        sim.save_m(incremental=True)
        sim.save_m(incremental=True)
        sim.save_m(incremental=True)
        assert(len(glob('barmini_m_[0-9]*.npy')) == 3)

        # Check that the 'overwrite' keyword works
        sim2 = barmini()
        with pytest.raises(IOError):
            sim2.save_m(incremental=True)
        sim2.save_m(incremental=True, overwrite=True)
        sim2.save_m(incremental=True)
        assert(len(glob('barmini_m_[0-9]*.npy')) == 2)

    def test_save_field_scheduled(self, tmpdir):
        os.chdir(str(tmpdir))
        sim = barmini()
        sim.schedule('save_field', 'm', every=1e-12)
        sim.run_until(2.5e-12)
        assert(len(glob('barmini_m_[0-9]*.npy')) == 3)
        sim.run_until(5.5e-12)
        assert(len(glob('barmini_m_[0-9]*.npy')) == 6)

        sim.clear_schedule()
        sim.schedule('save_field', 'm', filename='mag.npy', every=1e-12)
        sim.run_until(7.5e-12)
        assert(len(glob('barmini_m_[0-9]*.npy')) == 6)
        assert(len(glob('mag_[0-9]*.npy')) == 3)

    def test_sim_sllg(self, do_plot=False):
        mesh = df.BoxMesh(0, 0, 0, 2, 2, 2, 1, 1, 1)
        sim = Simulation(mesh, 8.6e5, unit_length=1e-9, kernel='sllg')
        alpha=0.1
        sim.alpha = alpha
        sim.set_m((1, 0, 0))
        sim.T = 0


        H0 = 1e5
        sim.add(Zeeman((0, 0, H0)))

        dt = 1e-12; ts = np.linspace(0, 500 * dt, 100)

        precession_coeff = sim.gamma / (1 + alpha ** 2)
        mz_ref = []

        mz = []
        real_ts=[]
        for t in ts:
            sim.run_until(t)
            real_ts.append(sim.t)
            mz_ref.append(np.tanh(precession_coeff * alpha * H0 * sim.t))
            mz.append(sim.m[-1]) # same as m_average for this macrospin problem

        mz=np.array(mz)

        if do_plot:
            import matplotlib.pyplot as plt
            ts_ns = np.array(real_ts) * 1e9
            plt.plot(ts_ns, mz, "b.", label="computed")
            plt.plot(ts_ns, mz_ref, "r-", label="analytical")
            plt.xlabel("time (ns)")
            plt.ylabel("mz")
            plt.title("integrating a macrospin")
            plt.legend()
            plt.savefig(os.path.join(MODULE_DIR, "test_sllg.png"))

        print("Deviation = {}, total value={}".format(
            np.max(np.abs(mz - mz_ref)),
            mz_ref))

        assert np.max(np.abs(mz - mz_ref)) < 8e-7

    def test_sim_sllg_time(self):
        mesh = df.BoxMesh(0, 0, 0, 5, 5, 5, 1, 1, 1)
        sim = Simulation(mesh, 8.6e5, unit_length=1e-9, kernel='sllg')
        sim.alpha = 0.1
        sim.set_m((1, 0, 0))
        sim.T = 10
        assert np.max(sim.T) == 10

        ts = np.linspace(0, 1e-9, 1001)

        H0 = 1e5
        sim.add(Zeeman((0, 0, H0)))

        real_ts=[]
        for t in ts:
            sim.run_until(t)
            real_ts.append(sim.t)

        print("Max Deviation = {}".format(
            np.max(np.abs(ts - real_ts))))

        assert np.max(np.abs(ts - real_ts)) < 1e-24


def test_sim_with(tmpdir):
    """
    Check that we can call sim_with with random values for all parameters.

    TODO: This test should arguably be more comprehensive.
    """
    os.chdir(str(tmpdir))
    mesh = df.UnitCubeMesh(3, 3, 3)
    demag_solver_params={'phi_1_solver': 'cg', 'phi_2_solver': 'cg', 'phi_1_preconditioner': 'ilu', 'phi_2_preconditioner': 'ilu'}
    sim = sim_with(mesh, Ms=8e5, m_init=[1, 0, 0], alpha=1.0, unit_length=1e-9, integrator_backend='sundials',
                   A=13e-12, K1=520e3, K1_axis=[0, 1, 1], H_ext=[0, 0, 1e6], D=6.98e-3, demag_solver='FK',
                   demag_solver_params=demag_solver_params, name='test_simulation')


def test_timezeeman_is_updated_automatically(tmpdir):
    """
    Check that the TimeZeeman.update() method is called automatically
    through sim.run_until() so that the field has the correct value at
    each time step.

    """
    def check_field_value(val):
        assert(np.allclose(H_ext.compute_field().reshape(3, -1).T, val, atol=0, rtol=1e-8))

    t_off = 3e-11
    t_end = 5e-11

    for method_name in ['run_until', 'advance_time']:
        sim = barmini()
        f = getattr(sim, method_name)

        field_expr = df.Expression(("0", "t", "0"), t=0)
        H_ext = TimeZeeman(field_expr, t_off=t_off)
        sim.add(H_ext)  # this should automatically register H_ext.update(), which is what we check next

        for t in np.linspace(0, t_end, 11):
            f(t)
            check_field_value([0, t, 0] if t < t_off else [0, 0, 0])


def test_ndt_writing_with_time_dependent_field(tmpdir):
    """
    Check that when we save time-dependent field values to a .ndt
    file, we actually write the values at the correct time steps (i.e.
    the ones requested by the scheduler and not the ones which are
    internally used by the time integrator).

    """
    os.chdir(str(tmpdir))
    TOL = 1e-8

    field_expr = df.Expression(("0", "t", "0"), t=0)
    H_ext = TimeZeeman(field_expr, t_off=2e-11)
    sim = barmini()
    sim.add(H_ext)
    sim.schedule('save_ndt', every=1e-12)
    sim.run_until(3e-11)

    Hy_expected = np.linspace(0, 3e-11, 31)
    Hy_expected[20:] = 0  # should be zero after the field was switched off

    f = Tablereader('barmini.ndt')
    assert np.allclose(f.timesteps(), np.linspace(0, 3e-11, 31), atol=0, rtol=TOL)
    assert np.allclose(f['H_TimeZeeman_x'], 0, atol=0, rtol=TOL)
    assert np.allclose(f['H_TimeZeeman_y'], Hy_expected, atol=0, rtol=TOL)
    assert np.allclose(f['H_TimeZeeman_z'], 0, atol=0, rtol=TOL)


def test_removing_logger_handlers_allows_to_create_many_simulation_objects(tmpdir):
    """
    When many simulation objects are created in the same scripts, the
    logger will eventually complain about 'too many open files'.
    Explicitly removing logger handlers should avoid this problem.

    """
    os.chdir(str(tmpdir))
    set_logging_level('WARNING')  # avoid lots of annoying info/debugging messages

    # Temporarily decrease the soft limit for the maximum number of
    # allowed open file descriptors (to make the test run faster and
    # ensure reproducibility across different machines).
    import resource
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (42, hard_limit))

    N = 150  # maximum number of simulation objects to create

    mesh = df.UnitIntervalMesh(1)
    Ms = 8e5
    unit_length = 1e-9

    def create_loads_of_simulations(N, close_logfiles=False):
        """
        Helper function to create lots of simulation objects,
        optionally closing previously created logfiles.

        """
        for i in xrange(N):
            logger.warning("Creating simulation object #{}".format(i))
            sim = Simulation(mesh, Ms, unit_length)
            if close_logfiles:
                sim.close_logfile()


    # The following should raise an error because lots of loggers are
    # created without being deleted again.
    with pytest.raises(IOError):
        create_loads_of_simulations(N, close_logfiles=False)

    # The next line is needed so that we can proceed after the error
    # raised above.
    default_timer.stop_last()

    # Remove all the file handlers created in the loop above
    hdls = list(logger.handlers)  # We need a copy of the list because we
                                  # are removing handlers from it below.
    for h in hdls:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            h.stream.close()  # this is essential, otherwise the file handler
                              # will remain open
            logger.removeHandler(h)

    # The following should work since we explicitly close the logfiles
    # before each Simulation object goes out of scope.
    create_loads_of_simulations(N, close_logfiles=True)

    # Check that no file logging handler is left
    print logging_status_str()

    # Restore the maximum number of allowed open file descriptors. Not
    # sure this is actually necessary but can't hurt.
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def test_schedule_render_scene(tmpdir):
    """
    Check that scheduling 'render_scene' will create incremental snapshots.

    Deactivated because it won't run on Jenkins without an X-Server.
    """
    os.chdir(str(tmpdir))
    sim = barmini()

    # Save the magnetisation using the default filename
    sim.schedule('render_scene', every=1e-11, filename='barmini_scene.png')
    sim.run_until(2.5e-11)
    assert(sorted(glob('barmini_scene_[0-9]*.png')) ==
           ['barmini_scene_000000.png',
            'barmini_scene_000001.png',
            'barmini_scene_000002.png'])


def test_sim_initialise_vortex(tmpdir, debug=True):
    """
    Call sim.initialise_vortex() for a cylindrical sample and a cuboid.
    If debug==True, a snapshots is saved for each of them for visual
    inspection.
    """
    os.chdir(str(tmpdir))
    mesh = nanodisk(d=60, h=5, maxh=3.0)
    sim = sim_with(mesh, Ms=8e6, m_init=[1, 0, 0], unit_length=1e-9)

    def save_debugging_output(sim, basename):
        if debug:
            sim.save_vtk(basename + '.pvd')
            sim.render_scene(outfile=basename + '.png')

    sim.initialise_vortex('simple', r=20)
    save_debugging_output(sim, 'disk_with_simple_vortex')

    # Vortex core is actually bigger than the sample but this shouldn't matter.
    sim.initialise_vortex('simple', r=40)
    save_debugging_output(sim, 'disk_with_simple_vortex2')

    # Try the Feldtkeller profile
    sim.initialise_vortex('feldtkeller', beta=15, center=(10, 0, 0), right_handed=False)
    save_debugging_output(sim, 'disk_with_feldtkeller_vortex')

    # Try a non-cylindrical sample, too, and optional arguments.
    sim = barmini()
    sim.initialise_vortex('simple', r=5, center=(2, 0, 0), right_handed=False)
    save_debugging_output(sim, 'barmini_with_vortex')


def test_sim_relax_accepts_filename(tmpdir):
    """
    Check that if sim.relax() is given a filename, the relaxed state
    is saved to this file.
    """
    sim = barmini()
    sim.set_m([1, 0, 0])
    sim.set_H_ext([1e6, 0, 0])
    sim.relax(save_restart_data_as='barmini_relaxed.npz',
              save_vtk_snapshot_as='barmini_relaxed.pvd',
              stopping_dmdt=10.0)
    assert(os.path.exists('barmini_relaxed.npz'))
    assert(os.path.exists('barmini_relaxed.pvd'))

    # Check that an existing file is  overwritten.
    os.remove('barmini_relaxed.pvd')
    sim.relax(save_restart_data_as='barmini_relaxed.npz',
              stopping_dmdt=10.0)
    assert(os.path.exists('barmini_relaxed.npz'))
    assert(not os.path.exists('barmini_relaxed.pvd'))


def test_NormalModeSimulation(tmpdir):
    os.chdir(str(tmpdir))
    nx = ny = nz = 2
    mesh = df.UnitCubeMesh(nx, ny, nz)
    sim = normal_mode_simulation(mesh, Ms=8e5, A=13e-12, m_init=[1, 0, 0], alpha=1.0, unit_length=1e-9, H_ext=[1e5, 1e3, 0], name='sim')
    sim.relax(stopping_dmdt=10.0)

    t_step = 1e-13
    sim.run_ringdown(t_end=1e-12, alpha=0.01, H_ext=[1e5, 0, 0], reset_time=True, save_ndt_every=t_step, save_m_every=t_step, m_snapshots_filename='foobar/foo_m.npy')

    assert(len(glob('foobar/foo_m*.npy')) == 11)
    f = Tablereader('sim.ndt')
    assert(np.allclose(f.timesteps(), np.linspace(0, 1e-12, 11), atol=0, rtol=1e-8))

    sim.reset_time(1.1e-12)  # hack to avoid a duplicate timestep at t=1e-12
    sim.run_ringdown(t_end=2e-12, alpha=0.02, H_ext=[1e4, 0, 0], reset_time=False, save_ndt_every=t_step, save_vtk_every=2*t_step, vtk_snapshots_filename='baz/sim_m.pvd')
    f.reload()
    assert(os.path.exists('baz/sim_m.pvd'))
    assert(len(glob('baz/sim_m*.vtu')) == 5)
    assert(np.allclose(f.timesteps(), np.linspace(0, 2e-12, 21), atol=0, rtol=1e-8))

    sim.plot_spectrum(use_averaged_m=True)
    sim.plot_spectrum(use_averaged_m=True, t_step=1.5e-12, subtract_values='first', figsize=(16, 6), filename='fft_m.png')
    # sim.plot_spectrum(use_averaged_m=False)
    # sim.plot_spectrum(use_averaged_m=True, t_step=1.5e-12, subtract_values='first', figsize=(16, 6), filename='fft_m_spatially_resolved.png')
    assert(os.path.exists('fft_m.png'))

    sim.plot_spectrum(t_step=t_step, filename='fft_m.png')

    sim.find_peak_near_frequency(10e9, component='y')

    sim.export_normal_mode_animation_from_ringdown('foobar/foo_m*.npy', peak_idx=2,
                                                   filename='animations/foo_peak_idx_2.pvd',
                                                   num_cycles=1, num_frames_per_cycle=4)
    sim.export_normal_mode_animation_from_ringdown('foobar/foo_m*.npy', f_approx=0.0, component='y',
                                                   directory='animations', num_cycles=1, num_frames_per_cycle=4)
    assert(os.path.exists('animations/foo_peak_idx_2.pvd'))
    assert(len(glob('animations/foo_peak_idx_2*.vtu')) == 4)

    # Either 'peak_idx' or both 'f_approx' and 'component' must be given
    with pytest.raises(ValueError):
        sim.export_normal_mode_animation_from_ringdown('foobar/foo_m*.npy', f_approx=0)
    with pytest.raises(ValueError):
        sim.export_normal_mode_animation_from_ringdown('foobar/foo_m*.npy', component='x')

    # Check that by default snapshots are not overwritten
    sim = normal_mode_simulation(mesh, Ms=8e5, A=13e-12, m_init=[1, 0, 0], alpha=1.0, unit_length=1e-9, H_ext=[1e5, 1e3, 0], name='sim')
    with pytest.raises(IOError):
        sim.run_ringdown(t_end=1e-12, alpha=0.02, H_ext=[1e4, 0, 0], save_vtk_every=2e-13, vtk_snapshots_filename='baz/sim_m.pvd')
    with pytest.raises(IOError):
        sim.run_ringdown(t_end=1e-12, alpha=0.02, H_ext=[1e4, 0, 0], save_m_every=2e-13, m_snapshots_filename='foobar/foo_m.npy')


def test_compute_normal_modes(tmpdir):
    os.chdir(str(tmpdir))

    d = 100
    h = 10
    maxh = 10.0
    alpha = 0.0
    m_init = [1, 0, 0]
    H_ext = [1e5, 0, 0]

    mesh = nanodisk(d, h, maxh)
    sim = normal_mode_simulation(mesh, Ms=8e6, m_init=m_init, alpha=alpha, unit_length=1e-9, A=13e-12, H_ext=H_ext, name='nanodisk')
    omega, w = sim.compute_normal_modes(n_values=10, filename_mat_A='matrix_A.npy', filename_mat_M='matrix_M.npy')
    print omega
    sim.export_normal_mode_animation(2, filename='animation/mode_2.pvd', num_cycles=1, num_snapshots_per_cycle=10, scaling=0.1)
    sim.export_normal_mode_animation(5, directory='animation', num_cycles=1, num_snapshots_per_cycle=10, scaling=0.1)

    assert(os.path.exists('animation/mode_2.pvd'))
    assert(len(glob('animation/mode_2*.vtu')) == 10)
    assert(len(glob('animation/normal_mode_5__*_GHz*.pvd')) == 1)
    assert(len(glob('animation/normal_mode_5__*_GHz*.vtu')) == 10)
