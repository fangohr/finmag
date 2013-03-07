import dolfin as df
import numpy as np
import logging
import pytest
import os
import shutil
from glob import glob
from tempfile import mkdtemp
from finmag import sim_with, Simulation
from finmag.example import barmini
from math import sqrt, cos, sin, pi
from finmag.util.helpers import assert_number_of_files,vector_valued_function
from finmag.sim import sim_helpers
from finmag.energies import Zeeman, Exchange, UniaxialAnisotropy

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
        a = np.loadtxt('barmini1.ndt')
        ta = a[:, 0]
        ta_expected = np.concatenate([np.linspace(0, 1e-12, 11),
                                      np.linspace(0.5e-12, 1.5e-12, 11)])
        assert(np.allclose(ta, ta_expected, atol=0))

        # Run a second simulation for 2 ps continuously
        sim2 = barmini(name='barmini2')
        sim2.schedule('save_ndt', every=1e-13)
        sim2.run_until(2e-12)

        # Check that the time steps for sim1 are as expected
        b = np.loadtxt('barmini2.ndt')
        tb = b[:, 0]
        tb_expected = np.linspace(0, 2e-12, 21)
        assert(np.allclose(tb, tb_expected, atol=0))

        # Check that the magnetisation dynamics of sim1 and sim2 are
        # the same and that we end up with the same magnetisation.
        a = np.concatenate([a[:10, :], a[11:, :]])  # delete the duplicate line due to the restart
        assert(np.allclose(a[:, 1:], b[:, 1:]))
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

        # Two different Zeeman interaction present
        sim.add(Zeeman((0, 0, 1)))
        sim.add(Zeeman((0, 0, 2)))
        with pytest.raises(ValueError):
            sim.remove_interaction("Zeeman")

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
        mesh = df.Box(0, 0, 0, 1, 1, 1, 1, 1, 1)
        sim = Simulation(mesh, Ms=8.6e5, unit_length=1e-9, name='macrospin_with_stt')
        sim.m = (1, 0, 0)
        sim.add(Zeeman([0, 0, 1e5]))
        sim.alpha = 0.0  # no damping

        def J(t):
            return 0.5e11 if (t < 2.5e-9) else -0.5e11

        sim.set_stt(0.05e11, 1.0, 2e-9, (0, 0, 1), with_time_update=J)
        sim.schedule('save_ndt', every=1e-11)
        sim.run_until(5e-9)

        ts, xs, ys, zs = np.loadtxt('macrospin_with_stt.ndt').T
        fig = plt.figure(figsize=(20, 5))
        ax1 = fig.add_subplot(131); ax1.plot(ts, xs)
        ax2 = fig.add_subplot(132); ax2.plot(ts, ys)
        ax3 = fig.add_subplot(133); ax3.plot(ts, zs)
        fig.savefig('macrospin_with_stt.png')

        # Assert that the dynamics of m_z are symmetric over time. In
        # theory, this should also be true of m_x and m_y, but since
        # they oscillate rapidly there is quite a bit of numerical
        # inaccuracy, so we're only testing for m_z here.
        assert max(abs(zs - zs[::-1])) < 0.001

    def test_mesh_info(self):
        mesh = df.Box(0, 0, 0, 1, 1, 1, 1, 1, 1)
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
