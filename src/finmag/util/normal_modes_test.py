from finmag.util.mesh_templates import Box, Sphere, Nanodisk
from finmag.util.meshes import mesh_info, plot_mesh_with_paraview
from finmag.util.normal_modes import *
from finmag import example
from finmag import sim_with, normal_mode_simulation
from math import pi
import logging
import pytest

logger = logging.getLogger("finmag")


def test_check_Kittel_mode_for_single_sphere(tmpdir):
    """
    Compute the eigenmodes of a perfect sphere and check that the
    frequency of the base mode equals the analytical value from the
    Kittel equation (see Charles Kittel, "Introduction to solid state
    physics", 7th edition, Ch. 16, p.505):

       omega_0 = gamma * B_0

    Here omega_0 is the angular frequency, so the actual frequency is
    equal to omega_0 / (2*pi).

    """
    os.chdir(str(tmpdir))

    sphere = Sphere(r=11)
    mesh = sphere.create_mesh(maxh=2.0)
    print "[DDD] mesh: {}".format(mesh)
    print mesh_info(mesh)
    plot_mesh_with_paraview(mesh, outfile='mesh_sphere.png')

    H_z = 4.42e5  # slightly odd value to make the test a bit more reliable
    frequency_unit = 1e9

    sim = sim_with(mesh, Ms=1e6, m_init=[0, 0, 1], A=13e-12, H_ext=[0, 0, H_z], unit_length=1e-9, demag_solver='FK')
    sim.relax()

    A, M, _, _ = compute_generalised_eigenproblem_matrices(sim, alpha=0.0, frequency_unit=1e9)

    RTOL = 1e-3
    n_values = 2
    n_values_export = 0
    omega, w = compute_normal_modes_generalised(A, M, n_values=n_values, discard_negative_frequencies=False)
    assert(len(omega) == n_values)

    # Check that the frequency of the base mode equals the analytical value from the Kittel equation
    # (see Charles Kittel, "Introduction to solid state physics", 7th edition, Ch. 16, p.505):
    #
    #    omega_0 = gamma * B_0
    #
    # The frequency is equal to omega_0 / (2*pi).
    #
    freq_expected = gamma * H_z / (2*pi*frequency_unit)
    assert(np.allclose(omega[0], +freq_expected, atol=0, rtol=RTOL))
    assert(np.allclose(omega[1], -freq_expected, atol=0, rtol=RTOL))
    logger.debug("Computed eigenfrequencies: {}".format(omega))
    logger.debug("Expected frequency of the Kittel mode: {}".format(freq_expected))

    # Perform the same test when negative frequencies are discarded
    omega_positive, _ = compute_normal_modes_generalised(A, M, n_values=n_values, discard_negative_frequencies=True)
    logger.debug("[DDD] omega_positive: {}".format(omega_positive))
    assert(len(omega_positive) == n_values)
    assert(np.allclose(omega_positive[0], freq_expected, atol=0, rtol=RTOL))

    # Ensure that the frequencies are all positive and sorted by absolute value
    assert((np.array(omega_positive) > 0).all())
    assert(np.allclose(omega_positive, sorted(omega_positive, key=abs)))

    omega_positive_first_half = omega_positive[:(n_values // 2)]
    assert(np.allclose(sorted(np.concatenate([omega_positive_first_half, -omega_positive_first_half])),
                       sorted(omega)))

    # Export normal mode animations for debugging
    for i in xrange(n_values_export):
        freq = omega_positive[i]
        export_normal_mode_animation(sim, freq, w[:, i], filename='normal_mode_{:02d}__{:.3f}_GHz.pvd'.format(i, freq))


    ##
    ## Perform the same test as above but without demag, wich should improve the accuracy.
    ##
    sim = sim_with(mesh, Ms=1e6, m_init=[0, 0, 1], A=13e-12, H_ext=[0, 0, H_z], unit_length=1e-9, demag_solver=None)
    sim.relax()

    A, M, _, _ = compute_generalised_eigenproblem_matrices(sim, alpha=0.0, frequency_unit=1e9)

    RTOL = 1e-9
    omega, _ = compute_normal_modes_generalised(A, M, n_values=1, discard_negative_frequencies=True)

    logger.debug("Computed eigenfrequencies (without demag): {}".format(omega))
    logger.debug("Expected frequency of the Kittel mode: {}".format(freq_expected))
    assert(np.allclose(omega[0], freq_expected, atol=0, rtol=RTOL))


def test_passing_scipy_eigsh_parameters(tmpdir):
    os.chdir(str(tmpdir))
    sim = example.normal_modes.disk()

    omega1, _ = sim.compute_normal_modes(n_values=4, tol=0)
    omega2, _ = sim.compute_normal_modes(n_values=4, tol=0, ncv=20, maxiter=2000, sigma=0.0, which='SM')
    logger.debug("Note: the following results are not meant to coincide! Their purpose is just to test passing arguments to scipy.sparse.linalg.eigsh")
    logger.debug("Computed eigenfrequencies #1: {}".format(omega1))
    logger.debug("Computed eigenfrequencies #2: {}".format(omega2))


def test_plot_spatially_resolved_normal_mode(tmpdir):
    os.chdir(str(tmpdir))
    d = 60
    h = 2
    sep = 5
    maxh = 10.0

    # Create a mesh consisting of two separate regions to check
    # whether matplotlib can handle the space in between them
    # correctly (previously it would connect the regions).
    center1 = (-0.5 * (d + sep), 0, 0)
    center2 = (+0.5 * (d + sep), 0, 0)
    nanodisk1 = Nanodisk(d, h, center=center1, name='nanodisk1')
    nanodisk2 = Nanodisk(d, h, center=center2, name='nanodisk2')
    nanodisks = nanodisk1 + nanodisk2
    mesh = nanodisks.create_mesh(maxh, directory='meshes')

    # Material parameters for Permalloy
    Ms = 8e5
    A=13e-12
    m_init = [1, 0, 0]
    alpha_relax = 1.0
    H_ext_relax = [1e4, 0, 0]

    sim = normal_mode_simulation(mesh, Ms, m_init, alpha=alpha_relax, unit_length=1e-9, A=A, H_ext=H_ext_relax)

    N = 3
    omega, eigenvecs = sim.compute_normal_modes(n_values=N)
    logger.debug("[DDD] Computed {} eigenvalues and {} eigenvectors.".format(len(omega), len(eigenvecs[0])))
    for i in xrange(N):
        #sim.export_normal_mode_animation(i, filename='animations/normal_mode_{:02d}/normal_mode_{:02d}.pvd'.format(i, i))
        w = eigenvecs[:, i]

        fig = plot_spatially_resolved_normal_mode(sim, w, slice_z='z_max', components='xyz',
                                                  figure_title='Eigenmodes', yshift_title=0.0,
                                                  plot_powers=True, plot_phases=True,
                                                  show_axis_labels=True, show_colorbars=True,
                                                  show_axis_frames=True, figsize=(12, 4),
                                                  outfilename='normal_mode_profile_{:02d}_v1.png'.format(i),
                                                  dpi=300)

        # Different combination of parameters
        fig = plot_spatially_resolved_normal_mode(sim, w, slice_z='z_min', components='xz',
                                                  plot_powers=True, plot_phases=False,
                                                  show_axis_frames=False, show_axis_labels=False,
                                                  show_colorbars=False, figsize=(12, 4),
                                                  outfilename='normal_mode_profile_{:02d}_v2.png'.format(i))

        with pytest.raises(ValueError):
            plot_spatially_resolved_normal_mode(sim, w, plot_powers=False, plot_phases=False)


def test_plot_spatially_resolved_normal_mode2(tmpdir):
    os.chdir(str(tmpdir))
    sim = example.normal_modes.disk()
    sim.compute_normal_modes(n_values=2)

    sim.plot_spatially_resolved_normal_mode(0, outfilename='normal_mode_00.png')
    sim.plot_spatially_resolved_normal_mode(1, slice_z='z_min', components='xz',
                                            figure_title='Title', yshift_title=0.05,
                                            plot_powers=True, plot_phases=False, num_phase_colorbar_ticks=3,
                                            cmap_powers=plt.cm.jet, cmap_phases=plt.cm.hsv, vmin_powers=None,
                                            show_axis_labels=True, show_axis_frames=True, show_colorbars=True, figsize=(5, 3),
                                            outfilename='normal_mode_01.png', dpi=200)

    assert(os.path.exists('normal_mode_00.png'))
    assert(os.path.exists('normal_mode_01.png'))


def test_is_hermitian():
    A = np.array([[1, 2j, 3+4j], [-2j, 1.3, 5-6j], [3-4j, 5+6j, 1.7]])
    B = np.array([[1, 2j, 3+4.0001j], [-2j, 1.3, 5-6j], [3-4j, 5+6j, 1.7]])

    assert(is_hermitian(A))
    assert(not is_hermitian(B))
    assert(is_hermitian(B, atol=1e-2))


if __name__ == '__main__':
    test_compute_generalised_eigenproblem_matrices_single_sphere('.')
