import os
import re
import types
import logging
import numpy as np
import dolfin as df
import matplotlib.pyplot as plt
from finmag.sim.sim import Simulation, sim_with
from finmag.normal_modes.eigenmodes import eigensolvers
from finmag.util import helpers
from finmag.util.fft import \
    compute_power_spectral_density, find_peak_near_frequency, _plot_spectrum, \
    export_normal_mode_animation_from_ringdown
from finmag.normal_modes.deprecated.normal_modes_deprecated import \
    compute_eigenproblem_matrix, compute_generalised_eigenproblem_matrices, \
    export_normal_mode_animation, plot_spatially_resolved_normal_mode

log = logging.getLogger(name="finmag")


class NormalModeSimulation(Simulation):
    """
    Thin wrapper around the Simulation class to make normal mode
    computations using the ringdown method more convenient.

    """
    def __init__(self, *args, **kwargs):
        super(NormalModeSimulation, self).__init__(*args, **kwargs)
        # Internal variables to store parameters/results of the ringdown method
        self.m_snapshots_filename = None
        self.t_step_ndt = None
        self.t_step_m = None

        # The following are used to cache computed spectra (potentially for various mesh regions)
        self.psd_freqs = {}
        self.psd_mx = {}
        self.psd_my = {}
        self.psd_mz = {}

        # Internal variables to store parameters/results of the (generalised) eigenvalue method
        self.eigenfreqs = None
        self.eigenvecs = None

        # Internal variables to store the matrices which define the generalised eigenvalue problem
        self.A = None
        self.M = None
        self.D = None
        self.use_real_matrix = None  # XXX TODO: Remove me once we get rid of the option 'use_real_matrix'
                                     #           in the method 'compute_normal_modes()' below.

        # Define a few eigensolvers which can be conveniently accesses using strings
        self.predefined_eigensolvers = {
            'scipy_dense': eigensolvers.ScipyLinalgEig(),
            'scipy_sparse': eigensolvers.ScipySparseLinalgEigs(sigma=0.0, which='LM'),
            'slepc_krylovschur': eigensolvers.SLEPcEigensolver(
                problem_type='GNHEP', method_type='KRYLOVSCHUR', which='SMALLEST_MAGNITUDE')
            }


    def run_ringdown(self, t_end, alpha, H_ext=None, reset_time=True, clear_schedule=True,
                     save_ndt_every=None, save_vtk_every=None, save_m_every=None,
                     vtk_snapshots_filename=None, m_snapshots_filename=None,
                     overwrite=False):
        """
        Run the ringdown phase of a normal modes simulation, optionally saving
        averages, vtk snapshots and magnetisation snapshots to the respective
        .ndt, .pvd and .npy files. Note that by default existing snapshots will
        not be overwritten. Use `overwrite=True` to achieve this.
        Also note that `H_ext=None` has the same effect as `H_ext=[0, 0, 0]`, i.e.
        any existing external field will be switched off during the ringdown phase.
        If you would like to have a field applied during the ringdown, you need to
        explicitly provide it.

        This function essentially wraps up the re-setting of parameters such as
        the damping value, the external field and the scheduled saving of data
        into a single convenient function call, thus making it less likely to
        forget any settings.

        The following snippet::

            sim.run_ringdown(t_end=10e-9, alpha=0.02, H_ext=[1e5, 0, 0],
                            save_m_every=1e-11, m_snapshots_filename='sim_m.npy',
                            save_ndt_every=1e-12)

        is equivalent to::

            sim.clear_schedule()
            sim.alpha = 0.02
            sim.reset_time(0.0)
            sim.set_H_ext([1e5, 0, 0])
            sim.schedule('save_ndt', every=save_ndt_every)
            sim.schedule('save_vtk', every=save_vtk_every, filename=vtk_snapshots_filename)
            sim.run_until(10e-9)

        """
        if reset_time:
            self.reset_time(0.0)
        if clear_schedule:
            self.clear_schedule()

        self.alpha = alpha
        if H_ext == None:
            if self.has_interaction('Zeeman'):
                log.warning("Found existing Zeeman field (for the relaxation "
                            "stage). Switching it off for the ringdown. "
                            "If you would like to keep it, please specify the "
                            "argument 'H_ext' explicitly.")
                self.set_H_ext([0, 0, 0])
        else:
            self.set_H_ext(H_ext)
        if save_ndt_every:
            self.schedule('save_ndt', every=save_ndt_every)
            log.debug("Setting self.t_step_ndt = {}".format(save_ndt_every))
            self.t_step_ndt = save_ndt_every

        def schedule_saving(which, every, filename, default_suffix):
            try:
                dirname, basename = os.path.split(filename)
                if dirname != '' and not os.path.exists(dirname):
                    os.makedirs(dirname)
            except AttributeError:
                dirname = os.curdir
                basename = self.name + default_suffix
            outfilename = os.path.join(dirname, basename)
            self.schedule(which, every=every, filename=outfilename, overwrite=overwrite)

            if which == 'save_m':
                # Store the filename so that we can later compute normal modes more conveniently
                self.m_snapshots_filename = outfilename

        if save_vtk_every:
            schedule_saving('save_vtk', save_vtk_every, vtk_snapshots_filename, '_m.pvd')

        if save_m_every:
            schedule_saving('save_m', save_m_every, m_snapshots_filename, '_m.npy')
            log.debug("Setting self.t_step_m = {}".format(save_m_every))
            self.t_step_m = save_m_every
            self.t_ini_m = self.t
            self.t_end_m = t_end

        self.run_until(t_end)

    def _compute_spectrum(self, use_averaged_m=False, mesh_region=None, **kwargs):
        try:
            if self.psd_freqs[mesh_region, use_averaged_m] != None and \
                    self.psd_mx[mesh_region, use_averaged_m] != None and \
                    self.psd_my[mesh_region, use_averaged_m] != None and \
                    self.psd_mz[mesh_region, use_averaged_m] != None:
                # We can use the cached results since the spectrum was computed before.
                return
        except KeyError:
            # No computations for these values of (mesh_region,
            # use_averaged_m) have been performed before. We're going
            # to do them below.
            pass

        if use_averaged_m:
            log.warning("Using the averaged magnetisation to compute the "
                        "spectrum is not recommended because this is likely "
                        "to miss certain symmetric modes.")
            if mesh_region != None:
                # TODO: we might still be able to compute this if the
                # user saved the averaged magnetisation in the
                # specified region to the .ndt file. We should check that.
                log.warning("Ignoring argument 'mesh_region' because "
                            "'use_averaged_m' was set to True.")
            filename = self.ndtfilename
        else:
            if self.m_snapshots_filename is None:
                log.warning(
                    "The spatially averaged magnetisation was not saved "
                    "during run_ringdown, thus it cannot be used to compute "
                    "the spectrum. Falling back to using the averaged "
                    "magnetisation (which is not recommended because it "
                    "is likely to miss normal modes which have certain "
                    "symmetries!).")
                if mesh_region != None:
                    log.warning("Ignoring argument 'mesh_region' because the "
                                "spatially resolved magnetisation was not "
                                "saved during the ringdown.")
                filename = self.ndtfilename
            else:
                # Create a wildcard pattern so that we can read the files using 'glob'.
                filename = re.sub('\.npy$', '*.npy', self.m_snapshots_filename)
        log.debug("Computing normal mode spectrum from file(s) '{}'.".format(filename))

        # Derive a sensible value of t_step, t_ini and t_end. Use the value
        # in **kwargs if one was provided, or else the one specified (or
        # derived) during a previous call of run_ringdown().
        t_step = kwargs.pop('t_step', None)#
        t_ini = kwargs.pop('t_ini', None)#
        t_end = kwargs.pop('t_end', None)#
        if t_step == None:
            if use_averaged_m:
                t_step = self.t_step_ndt
            else:
                t_step = self.t_step_m
        if t_ini == None and not use_averaged_m:
                t_ini = self.t_ini_m
        if t_end == None and not use_averaged_m:
                t_end = self.t_end_m
        # XXX TODO: Use t_step_npy if computing the spectrum from .npy files.
        if t_step == None:
            raise ValueError(
                "No sensible default for 't_step' could be determined. "
                "(It seems like 'run_ringdown()' was not run, or it was not "
                "given a value for its argument 'save_ndt_every'). Please "
                "call sim.run_ringdown() or provide the argument 't_step' "
                "explicitly.")

        if mesh_region != None:
            # If a mesh region is specified, restrict computation of the
            # spectrum to the corresponding mesh vertices.
            submesh = self.get_submesh(mesh_region)
            try:
                # Legacy syntax (for dolfin <= 1.2 or so).
                # TODO: This should be removed in the future once dolfin 1.3 is released!
                parent_vertex_indices = submesh.data().mesh_function('parent_vertex_indices').array()
            except RuntimeError:
                # This is the correct syntax now, see:
                # http://fenicsproject.org/qa/185/entity-mapping-between-a-submesh-and-the-parent-mesh
                parent_vertex_indices = submesh.data().array('parent_vertex_indices', 0)
            kwargs['restrict_to_vertices'] = parent_vertex_indices

        psd_freqs, psd_mx, psd_my, psd_mz = \
            compute_power_spectral_density(filename, t_step=t_step, t_ini=t_ini, t_end=t_end, **kwargs)

        self.psd_freqs[mesh_region, use_averaged_m] = psd_freqs
        self.psd_mx[mesh_region, use_averaged_m] = psd_mx
        self.psd_my[mesh_region, use_averaged_m] = psd_my
        self.psd_mz[mesh_region, use_averaged_m] = psd_mz

    def plot_spectrum(self, t_step=None, t_ini=None, t_end=None, subtract_values='first',
                      components="xyz", log=False, xlim=None, ticks=5, figsize=None, title="",
                      outfilename=None, use_averaged_m=False, mesh_region=None):
        """
        Plot the normal mode spectrum of the simulation. If
        `mesh_region` is not None, restricts the computation to mesh
        vertices in that region (which must have been specified with
        `sim.mark_regions` before).

        This is a convenience wrapper around the function
        finmag.util.fft.plot_power_spectral_density. It accepts the
        same keyword arguments, but provides sensible defaults for
        some of them so that it is more convenient to use. For
        example, `ndt_filename` will be the simulation's .ndt file by
        default, and t_step will be taken from the value of the
        argument `save_ndt_every` when sim.run_ringdown() was run.

        The default method to compute the spectrum is the one described
        in [1], which uses a spatially resolved Fourier transform
        to compute the local power spectra and then integrates these over
        the sample (which yields the total power over the sample at each
        frequency). This is the recommended way to compute the spectrum,
        but it requires the spatially resolved magnetisation to be saved
        during the ringdown (provide the arguments `save_m_every` and
        `m_snapshots_filename` to `sim.run_ringdown()` to achieve this).

        An alternative method (used when `use_averaged_m` is True) simply
        computes the Fourier transform of the spatially averaged magnetisation
        and plots that. However, modes with certain symmetries (which are
        quite common) will not be detected by this method, so it is not
        recommended to use it. It will be used as a fallback, however,
        if the spatially resolved magnetisation was not saved during the
        ringdown phase.

        [1] McMichael, Stiles, "Magnetic normal modes of nanoelements", J Appl Phys 97 (10), 10J901, 2005.

        """
        self._compute_spectrum(t_step=t_step, t_ini=t_ini, t_end=t_end, subtract_values=subtract_values, use_averaged_m=use_averaged_m, mesh_region=mesh_region)

        fig = _plot_spectrum(self.psd_freqs[mesh_region, use_averaged_m],
                             self.psd_mx[mesh_region, use_averaged_m],
                             self.psd_my[mesh_region, use_averaged_m],
                             self.psd_mz[mesh_region, use_averaged_m],
                             components=components, log=log, xlim=xlim,
                             ticks=ticks, figsize=figsize, title=title,
                             outfilename=outfilename)
        return fig

    def _get_psd_component(self, component, mesh_region, use_averaged_m):
        try:
            res = {'x': self.psd_mx[mesh_region, use_averaged_m],
                   'y': self.psd_my[mesh_region, use_averaged_m],
                   'z': self.psd_mz[mesh_region, use_averaged_m]
                   }[component]
        except KeyError:
            raise ValueError("Argument `component` must be exactly one of 'x', 'y', 'z'.")
        return res

    def find_peak_near_frequency(self, f_approx, component=None, use_averaged_m=False, mesh_region=None):
        """
        XXX TODO: Write me!

        The argument `use_averaged_m` has the same meaning as in the `plot_spectrum`
        method. See its documentation for details.
        """
        if component is None:
            log.warning("Please specify the 'component' argument (which "
                        "determines the magnetization component in which "
                        "to search for a peak.")
            return

        if f_approx is None:
            raise TypeError("Argument 'f_approx' must not be None.")
        if not isinstance(component, types.StringTypes):
            raise TypeError("Argument 'component' must be of type string.")

        self._compute_spectrum(use_averaged_m=use_averaged_m, mesh_region=mesh_region)
        psd_cmpnt = self._get_psd_component(component, mesh_region, use_averaged_m)

        return find_peak_near_frequency(f_approx, self.psd_freqs[mesh_region, use_averaged_m], psd_cmpnt)

    def find_all_peaks(self, component, use_averaged_m=False, mesh_region=None):
        """
        Return a list all peaks in the spectrum of the given magnetization component.

        """
        self._compute_spectrum(use_averaged_m=use_averaged_m, mesh_region=mesh_region)
        freqs = self.psd_freqs[mesh_region, use_averaged_m]
        all_peaks = sorted(list(set([self.find_peak_near_frequency(x, component=component, use_averaged_m=use_averaged_m, mesh_region=mesh_region) for x in freqs])))
        return all_peaks

    def plot_peak_near_frequency(self, f_approx, component, mesh_region=None, use_averaged_m=False, **kwargs):
        """
        Convenience function for debugging which first finds a peak
        near the given frequency and then plots the spectrum together
        with a point marking the detected peak.

        Internally, this calls `sim.find_peak_near_frequency` and
        `sim.plot_spectrum()` and accepts all keyword arguments
        supported by these two functions.

        """
        peak_freq, peak_idx = self.find_peak_near_frequency(f_approx, component, mesh_region=mesh_region)
        psd_cmpnt = self._get_psd_component(component, mesh_region, use_averaged_m)
        fig = self.plot_spectrum(use_averaged_m=use_averaged_m, **kwargs)
        fig.gca().plot(self.psd_freqs[mesh_region, use_averaged_m][peak_idx] / 1e9, psd_cmpnt[peak_idx], 'bo')
        return fig

    def export_normal_mode_animation_from_ringdown(self, npy_files, f_approx=None, component=None,
                                                   peak_idx=None, outfilename=None, directory='',
                                                   t_step=None, scaling=0.2, dm_only=False,
                                                   num_cycles=1, num_frames_per_cycle=20,
                                                   use_averaged_m=False):
        """
        XXX TODO: Complete me!

        If the exact index of the peak in the FFT array is known, e.g.
        because it was computed via `sim.find_peak_near_frequency()`,
        then this can be given as the argument `peak_index`.
        Otherwise, `component` and `f_approx` must be given and these
        are passed on to `sim.find_peak_near_frequency()` to determine
        the exact location of the peak.

        The output filename can be specified via `outfilename`. If this
        is None then a filename of the form 'normal_mode_N__xx.xxx_GHz.pvd'
        is generated automatically, where N is the peak index and xx.xx
        is the frequency of the peak (as returned by the function
        `sim.find_peak_near_frequency()`).


        *Arguments*:

        f_approx:  float

           Find and animate peak closest to this frequency. This is
           ignored if 'peak_idx' is given.

        component:  str

           The component (one of 'x', 'y', 'z') of the FFT spectrum
           which should be searched for a peak. This is ignored if
           peak_idx is given.

        peak_idx:  int

           The index of the peak in the FFT spectrum. This can be
           obtained by calling `sim.find_peak_near_frequency()`.
           Alternatively, if the arguments `f_approx` and `component`
           are given then this index is computed automatically. Note
           that if `peak_idx` is given the other two arguments are
           ignored.

        use_averaged_m:  bool

           Determines the method used to compute the spectrum. See
           the method `plot_spectrum()` for details.

        """
        self._compute_spectrum(use_averaged_m=use_averaged_m)

        if peak_idx is None:
            if f_approx is None or component is None:
                raise ValueError("Please specify either 'peak_idx' or both 'f_approx' and 'component'.")
            peak_freq, peak_idx = self.find_peak_near_frequency(f_approx, component)
        else:
            if f_approx != None:
                log.warning("Ignoring argument 'f_approx' because 'peak_idx' was specified.")
            if component != None:
                log.warning("Ignoring argument 'component' because 'peak_idx' was specified.")
            peak_freq = self.psd_freqs[None, use_averaged_m][peak_idx]  # XXX TODO: Currently mesh regions are not supported, so we must use None here. Can we fix this?!?

        if outfilename is None:
            if directory is '':
                raise ValueError("Please specify at least one of the arguments 'outfilename' or 'directory'")
            outfilename = 'normal_mode_{}__{:.3f}_GHz.pvd'.format(peak_idx, peak_freq / 1e9)
        outfilename = os.path.join(directory, outfilename)

        if t_step == None:
            if (self.t_step_ndt != None) and (self.t_step_m != None) and \
                    (self.t_step_ndt != self.t_step_m):
                log.warning("Values of t_step for previously saved .ndt and .npy data differ! ({} != {}). Using t_step_ndt, but please double-check this is what you want.".format(self.t_step_ndt, self.t_step_m))
        t_step = t_step or self.t_step_ndt
        export_normal_mode_animation_from_ringdown(npy_files, outfilename, self.mesh, t_step,
                                                   peak_idx, dm_only=dm_only, num_cycles=num_cycles,
                                                   num_frames_per_cycle=num_frames_per_cycle)

    def assemble_eigenproblem_matrices(self, filename_mat_A=None, filename_mat_M=None,  use_generalized=False,
                                       force_recompute_matrices=False, check_hermitian=False,
                                       differentiate_H_numerically=True, use_real_matrix=True):
        if use_generalized:
            if (self.A == None or self.M == None) or force_recompute_matrices:
                df.tic()
                self.A, self.M, _, _ = compute_generalised_eigenproblem_matrices( \
                    self, frequency_unit=1e9, filename_mat_A=filename_mat_A, filename_mat_M=filename_mat_M,
                    check_hermitian=check_hermitian, differentiate_H_numerically=differentiate_H_numerically)
                log.debug("Assembling the eigenproblem matrices took {}".format(helpers.format_time(df.toc())))
            else:
                log.debug('Re-using previously computed eigenproblem matrices.')
        else:
            if self.D == None or (self.use_real_matrix != use_real_matrix) or force_recompute_matrices:
                df.tic()
                self.D = compute_eigenproblem_matrix(
                             self, frequency_unit=1e9, differentiate_H_numerically=differentiate_H_numerically,
                             dtype=(float if use_real_matrix else complex))
                self.use_real_matrix = use_real_matrix
                log.debug("Assembling the eigenproblem matrix took {}".format(helpers.format_time(df.toc())))
            else:
                log.debug('Re-using previously computed eigenproblem matrix.')

    def compute_normal_modes(self, n_values=10, solver='scipy_dense',
                             discard_negative_frequencies=True,
                             filename_mat_A=None, filename_mat_M=None,
                             use_generalized=False, force_recompute_matrices=False,
                             check_hermitian=False, differentiate_H_numerically=True,
                             use_real_matrix=True):
        """
        Compute the eigenmodes of the simulation by solving a generalised
        eigenvalue problem and return the computed eigenfrequencies and
        eigenvectors.  The simulation must be relaxed for this to yield
        sensible results.

        All keyword arguments not mentioned below are directly passed on
        to `scipy.sparse.linalg.eigs`, which is used internally to solve
        the generalised eigenvalue problem.


        *Arguments*

        n_values:

            The number of eigenmodes to compute (returns the
            `n_values` smallest ones).

        solver:

            The type of eigensolver to use. This should be an instance of
            one of the Eigensolver classes defined in the module
            `finmag.normal_modes.eigenmodes.eigensolvers`. For convenience,
            there are also some predefined options which can be accessed
            using the strings "scipy_dense", "scipy_sparse" and
            "slepc_krylovschur".

            Examples:

                ScipyLinalgEig()
                ScipySparseLinalgEigs(sigma=0.0, which='LM', tol=1e-8)
                SLEPcEigensolver(problem_type='GNHEP', method_type='KRYLOVSCHUR', which='SMALLEST_MAGNITUDE', tol=1e-8, maxit=400)

        discard_negative_frequencies:

            For every eigenmode there is usually a corresponding mode
            with the negative frequency which otherwise looks exactly
            the same (at least in sufficiently symmetric situations).
            If `discard_negative_frequencies` is True then these are
            discarded. Use this at your own risk, however, since you
            might miss genuinely different eigenmodes. Default is
            False.

        filename_mat_A:
        filename_mat_M:

            If given (default: None), the matrices A and M which
            define the generalised eigenvalue problem are saved to
            files with the specified names. This can be useful for
            debugging or later post-processing to avoid having to
            recompute these matrices.

        use_generalized:

            If True (default: False), compute the eigenmodes using the
            formulation as a generalised eigenvalue problem instead of
            an ordinary one.

        force_recompute_matrices:

            If False (the default), the matrices defining the
            generalised eigenvalue problem are not recomputed if they
            have been computed before (this dramatically speeds up
            repeated computation of eigenmodes). Set to True to
            force recomputation.

        check_hermitian:

            If True, check whether the matrix A used in the generalised
            eigenproblem is Hermitian and print a warning if this is not
            the case. This is a temporary debugging flag which is likely
            to be removed again in the future. Default value: False.

        differentiate_H_numerically:

            If True (the default), compute the derivative of the effective
            field numerically. If False, use a more efficient method. The
            latter is much faster, but leads to numerical inaccuracies so
            it is only recommended for testing large systems.

        use_real_matrix:

            This argument only applies if `use_generalised` is False. In this
            case, if `use_real_matrix` is True (the default), passes a matrix
            with real instead of complex entries to the solver. This is
            recommended as it reduces the computation time considerably.
            This option is for testing purposes only and will be removed in
            the future.


        *Returns*

        A triple (omega, eigenvectors, rel_errors), where omega is a list of
        the `n_values` smallest eigenfrequencies, `eigenvectors` is a rectangular
        array whose rows are the corresponding eigenvectors (in the same order),
        and rel_errors are the relative errors of each eigenpair, defined as
        follows:

            rel_err = ||A*v - omega*M*v||_2 / ||omega*v||_2

        NOTE: Previously, the eigenvectors would be returned in the *columns*
              of `w`, but this has changed in the interface!

        """
        if isinstance(solver, str):
            try:
                solver = self.predefined_eigensolvers[solver]
            except KeyError:
                raise ValueError("Unknown eigensolver: '{}'".format(solver))

        if use_generalized and isinstance(solver, eigensolvers.SLEPcEigensolver):
            raise TypeError("Using the SLEPcEigensolver with a generalised "
                            "eigenvalue problemis not currently implemented.")

        self.assemble_eigenproblem_matrices(
            filename_mat_A=filename_mat_A, filename_mat_M=filename_mat_M, use_generalized=use_generalized,
            force_recompute_matrices=force_recompute_matrices, check_hermitian=check_hermitian,
            differentiate_H_numerically=differentiate_H_numerically, use_real_matrix=use_real_matrix)

        if discard_negative_frequencies:
            # If negative frequencies should be discarded, we need to compute twice as many as the user requested
            n_values *= 2

        if use_generalized:
            # omega, eigenvecs = compute_normal_modes_generalised(self.A, self.M, n_values=n_values, discard_negative_frequencies=discard_negative_frequencies,
            #                                             tol=tol, sigma=sigma, which=which, v0=v0, ncv=ncv, maxiter=maxiter, Minv=Minv, OPinv=OPinv, mode=mode)
            omega, eigenvecs, rel_errors = solver.solve_eigenproblem(self.A, self.M, num=n_values)
        else:
            # omega, eigenvecs = compute_normal_modes(self.D, n_values, sigma=0.0, tol=tol, which='LM')
            # omega = np.real(omega)  # any imaginary part is due to numerical inaccuracies so we ignore them
            omega, eigenvecs, rel_errors = solver.solve_eigenproblem(self.D, None, num=n_values)
            if use_real_matrix:
                # Eigenvalues are complex due to the missing factor of 1j in the matrix with real entries. Here we correct for this.
                omega = 1j*omega

        # Sanity check: frequencies should occur in +/- pairs
        pos_freqs = filter(lambda x: x >= 0, omega)
        neg_freqs = filter(lambda x: x <= 0, omega)
        #kk = min(len(pos_freqs), len(neg_freqs))
        for (freq1, freq2) in zip(pos_freqs, neg_freqs):
            if not np.isclose(freq1.real, -freq2.real, rtol=1e-8):
                log.error("Frequencies should occur in +/- pairs, but computed: {:f} / {:f}".format(freq1, freq2))

        # Another sanity check: frequencies should be approximately real
        for freq in omega:
            if not abs(freq.imag / freq) < 1e-4:
                log.warning('Frequencies should be approximately real, but computed: {:f}'.format(freq))

        if discard_negative_frequencies:
            pos_freq_indices = [i for (i, freq) in enumerate(omega) if freq >= 0]
            omega = omega[pos_freq_indices]
            eigenvecs = eigenvecs[pos_freq_indices]

        log.debug("Relative errors of computed eigensolutions: {}".format(rel_errors))

        self.eigenfreqs = omega
        self.eigenvecs = eigenvecs
        self.rel_errors = rel_errors

        return omega, eigenvecs, rel_errors

    def export_eigenmode_animations(self, modes, dm_only=False, directory='', create_movies=True, directory_movies=None, num_cycles=1, num_snapshots_per_cycle=20, scaling=0.2, **kwargs):
        """
        Export animations for multiple eigenmodes.

        The first argument `modes` can be either an integer (to export
        the first N modes) or a list of integers (to export precisely
        specified modes).

        If `dm_only` is `False` (the default) then the "full" eigenmode
        of the form m0+dm(t) will be exported (where m0 is the relaxed
        equilibrium state and dm(t) is the small time-varying eigenmode
        excitation). If `dm_only` is `True` then only the part dm(t)
        will be exported.

        The animations will be saved in VTK format (as .pvd files) and
        stored in separate subfolders of the given `directory` (one
        subfolder for each eigenmode).

        If `create_movies` is `True` (the default) then each animation
        will in addition be automatically converted to a movie file
        in .avi format and stored in the directory `dirname_movies`.
        (If the latter is `None` (the default), then the movies will
        be stored in `directory` as well).

        This function accepts all keyword arguments understood by
        `finmag.util.visualization.render_paraview_scene()`, which
        can be used to control aspects such as the camera position
        or whether to add glyphs etc. in the exported movie files.

        """
        # Make sure that `modes` is a list of integers
        if not hasattr(modes, '__len__'):
            modes = range(modes)

        # Export eigenmode animations
        for k in modes:
            try:
                freq = self.eigenfreqs[k]
            except IndexError:
                log.error("Could not export eingenmode animation for mode "
                          "#{}. Index exceeds range of comptued modes.".format(k))
                continue

            mode_descr = 'normal_mode_{:02d}__{:.3f}_GHz'.format(k, freq)
            vtk_filename = os.path.join(directory, mode_descr, mode_descr + '.pvd')

            self.export_normal_mode_animation(
                k, filename=vtk_filename, dm_only=dm_only, num_cycles=num_cycles,
                num_snapshots_per_cycle=num_snapshots_per_cycle, scaling=scaling, **kwargs)

            # Some good default values for movie files
            default_kwargs = {
                'representation': 'Surface',
                'add_glyphs': True,
                'trim_border': False,
                'colormap': 'coolwarm',
                }
            # Update default values with the kwargs explicitly given by the user
            default_kwargs.update(kwargs)

            if create_movies:
                directory_movies = directory_movies or directory
                helpers.pvd2avi(vtk_filename, os.path.join(directory_movies, mode_descr + '.avi'), **default_kwargs)

    def export_normal_mode_animation(self, k, filename=None, directory='', dm_only=False, num_cycles=1, num_snapshots_per_cycle=20, scaling=0.2, **kwargs):
        """
        XXX TODO: Complete me!

        Export an animation of the `k`-th eigenmode, where the value of `k`
        refers to the index of the corresponding mode frequency and eigenvector
        in the two arrays returnd by `sim.compute_normal_modes`. If that method
        was called multiple times the results of the latest call are used.

        """
        if self.eigenfreqs is None or self.eigenvecs is None:
            log.debug("Could not find any precomputed eigenmodes. Computing them now.")
            self.compute_normal_modes(max(k, 10))

        if filename is None:
            if directory is '':
                raise ValueError("Please specify at least one of the arguments 'filename' or 'directory'")
            filename = 'normal_mode_{}__{:.3f}_GHz.pvd'.format(k, self.eigenfreqs[k])
        filename = os.path.join(directory, filename)

        basename, suffix = os.path.splitext(filename)

        # Export VTK animation
        if suffix == '.pvd':
            pvd_filename = filename
        elif suffix in ['.jpg', '.avi']:
            pvd_filename = basename + '.pvd'
        else:
            raise ValueError("Filename must end in one of the following suffixes: .pvd, .jpg, .avi.")

        export_normal_mode_animation(self.mesh, self.m, self.eigenfreqs[k], self.eigenvecs[k],
                                     pvd_filename, num_cycles=num_cycles,
                                     num_snapshots_per_cycle=num_snapshots_per_cycle,
                                     scaling=scaling, dm_only=dm_only)

        # Export image files
        if suffix == '.jpg':
            jpg_filename = filename
        elif suffix == '.avi':
            jpg_filename = basename + '.jpg'
        else:
            jpg_filename = None

        if jpg_filename != None:
            from finmag.util.visualization import render_paraview_scene
            render_paraview_scene(pvd_filename, outfile=jpg_filename, **kwargs)

        # Convert image files to movie
        if suffix == '.avi':
            movie_filename = filename
            helpers.pvd2avi(pvd_filename, outfile=movie_filename)

        # Return the movie if it was created
        res = None
        if suffix == '.avi':
            from IPython.display import HTML
            from base64 import b64encode
            video = open(movie_filename, "rb").read()
            video_encoded = b64encode(video)
            #video_tag = '<video controls alt="test" src="data:video/x-m4v;base64,{0}">'.format(video_encoded)
            #video_tag = '<a href="files/{0}">Link to video</a><br><br><video controls alt="test" src="data:video.mp4;base64,{1}">'.format(movie_filename, video_encoded)
            #video_tag = '<a href="files/{0}" target="_blank">Link to video</a><br><br><embed src="files/{0}"/>'.format(movie_filename)  # --> kind of works
            #video_tag = '<a href="files/{0}" target="_blank">Link to video</a><br><br><object data="files/{0}" type="video.mp4"/>'.format(movie_filename)  # --> kind of works
            #video_tag = '<a href="files/{0}" target="_blank">Link to video</a><br><br><embed src="files/{0}"/>'.format(basename + '.avi')  # --> kind of works

            # Display a link to the video
            video_tag = '<a href="files/{0}" target="_blank">Link to video</a>'.format(basename + '.avi')
            return HTML(data=video_tag)

    def plot_spatially_resolved_normal_mode(self, k, slice_z='z_max', components='xyz', region=None,
                                            figure_title=None, yshift_title=0.0,
                                            plot_powers=True, plot_phases=True, num_phase_colorbar_ticks=3,
                                            cmap_powers=plt.cm.jet, cmap_phases=plt.cm.hsv, vmin_powers=None,
                                            show_axis_labels=True, show_axis_frames=True,
                                            show_colorbars=True, figsize=None,
                                            outfilename=None, dpi=None, use_fenicstools=False):
        """
        Plot a spatially resolved profile of the k-th normal mode as
        computed by `sim.compute_normal_modes()`.

        *Returns*

        A `matplotlib.Figure` containing the plot.


        See the docstring of the function
        `finmag.normal_modes.deprecated.normal_modes.plot_spatially_resolved_normal_mode`
        for details about the meaning of the arguments.

        """
        if self.eigenvecs == None:
            log.warning("No eigenvectors have been computed. Please call "
                        "`sim.compute_normal_modes()` to do so.")

        if region == None:
            mesh = self.mesh
            w = self.eigenvecs[k]
            m = self.m
        else:
            # Restrict m and the eigenvector array to the submesh
            # TODO: This is messy and should be factored out into helper routines.
            mesh = self.get_submesh(region)
            restr = helpers.restriction(self.mesh, mesh)
            m = restr(self.m.reshape(3, -1)).ravel()
            w1, w2 = self.eigenvecs[k].reshape(2, -1)
            w1_restr = restr(w1)
            w2_restr = restr(w2)
            w = np.concatenate([w1_restr, w2_restr]).reshape(2, -1)

        fig = plot_spatially_resolved_normal_mode(
            mesh, m, w, slice_z=slice_z, components=components,
            figure_title=figure_title, yshift_title=yshift_title,
            plot_powers=plot_powers, plot_phases=plot_phases,
            cmap_powers=cmap_powers, cmap_phases=cmap_phases, vmin_powers=vmin_powers,
            show_axis_labels=show_axis_labels, show_axis_frames=show_axis_frames,
            show_colorbars=show_colorbars, figsize=figsize,
            outfilename=outfilename, dpi=dpi, use_fenicstools=use_fenicstools)
        return fig


def normal_mode_simulation(mesh, Ms, m_init, **kwargs):
    """
    Same as `sim_with` (it accepts the same keyword arguments apart from
    `sim_class`), but returns an instance of `NormalModeSimulation`
    instead of `Simulation`.

    """
    # Make sure we don't inadvertently create a different kind of simulation
    kwargs.pop('sim_class', None)

    return sim_with(mesh, Ms, m_init, sim_class=NormalModeSimulation, **kwargs)
