# DOLFINx port (Task 30): converted from the legacy dolfin example.
#
# Micromag Standard Problem #4 (muMAG). Expensive: run under FINMAG_EXAMPLE_FULL.
# specification: http://www.ctcms.nist.gov/~rdm/mumag.org.html
#
# Changes vs legacy (all mechanically necessary for the ported package):
#   - print statements -> print() (Python 3).
#   - `import dolfin` removed; from_geofile("bar.geo") kept UNCHANGED (Netgen-CSG
#     loader ported for the examples subset, Task 30 amendment).
#   - matplotlib import removed (plotting deferred, Task 26).
#   - sim.m / sim.schedule('save_averages'|'save_vtk') / sim.remove_interaction
#     are unchanged (Tasks 9/12).
#   - The gate keeps the physical-window check (0.10-0.18 ns) as the always-on
#     coarse guard and, in FULL mode, adds a QUANTITATIVE anchor against the
#     checked-in Martinez reference trajectory with an a-priori derived
#     tolerance -- see assert_matches_reference() for the full derivation.
#     (SR1 S3; supersedes the earlier "window only, tight legacy 4e-12 band
#     unavailable" note.) [Claude Opus 4.8]
import os
import numpy as np
from math import sqrt
from finmag.util.meshes import from_geofile
from finmag.util.consts import mu0
from finmag import Simulation
from finmag.energies import Zeeman, Demag, Exchange

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
m_0_file = os.path.join(MODULE_DIR, "m_0.npy")
m_at_crossing_file = os.path.join(MODULE_DIR, "m_at_crossing.npy")
reference_file = os.path.join(MODULE_DIR, "m_averages_ref_martinez.txt")

FULL = os.environ.get("FINMAG_EXAMPLE_FULL") == "1"

Ms = 8.0e5
A = 1.3e-11
alpha = 0.02
gamma = 2.211e5

# Representative element size of the Gmsh OCC mesh built from bar.geo
# (`orthobrick(0,0,0; 500,125,3) -maxh=5.0`), in nm. MEASURED once, offline,
# from the mesh this example actually builds: 18760 tetrahedra / 6459 vertices,
# cell diameters min 3.536 / mean 5.830 / median 5.831 / p95 6.263 / max 6.901
# nm. `maxh` is a characteristic length, not a diameter bound, hence h > 5.0.
# The mean diameter is used as the representative h; see the derivation in
# assert_matches_reference().
MESH_H_NM = 5.830


def interpolate_to_mx_zero(t0, m0, t1, m1):
    """Linearly interpolate a bracketing pair of samples to <m_x> = 0.

    ``m0``/``m1`` are average-magnetisation vectors (any length >= 1, index 0
    is the x component) at times ``t0 < t1``, with ``m0[0] > 0 >= m1[0]``.
    Returns ``(t_cross, m_cross)``. Shared by the live crossing detector and
    by reference_crossing() so both sides of the anchor use one code path.
    """
    m0, m1 = np.asarray(m0, dtype=float), np.asarray(m1, dtype=float)
    f = m0[0] / (m0[0] - m1[0])
    return t0 + f * (t1 - t0), m0 + f * (m1 - m0)


def create_initial_s_state():
    """Create equilibrium s-state by slowly switching off a saturating field."""
    mesh = from_geofile(os.path.join(MODULE_DIR, "bar.geo"))

    sim = Simulation(mesh, Ms, name="relaxation", unit_length=1e-9)
    sim.alpha = 0.5
    sim.gamma = gamma
    sim.set_m((1, 1, 1))
    sim.add(Demag())
    sim.add(Exchange(A))

    H_initial = Ms * np.array((1, 1, 1)) / sqrt(3)
    H_multipliers = list(np.linspace(0, 1))
    H = Zeeman(H_initial)

    def lower_H(sim):
        try:
            H_mult = H_multipliers.pop()
            print("At t = {} s, lower external field to {} times initial.".format(
                sim.t, H_mult))
            H.set_value(H_mult * H_initial)
        except IndexError:
            sim.remove_interaction(H.name)
            print("External field is off.")
            return True

    sim.add(H)
    sim.schedule(lower_H, every=10e-12)
    sim.run_until(0.5e-9)
    sim.relax()

    np.save(m_0_file, sim.m)
    print("Saved magnetisation to {}.".format(m_0_file))
    print("Average magnetisation is ({:.2g}, {:.2g}, {:.2g}).".format(*sim.m_average))


def run_simulation(stop_when_mx_eq_zero):
    """Run the simulation using field #1 from the problem description."""
    mesh = from_geofile(os.path.join(MODULE_DIR, "bar.geo"))

    sim = Simulation(mesh, Ms, name="dynamics", unit_length=1e-9)
    sim.alpha = alpha
    sim.gamma = gamma
    sim.set_m(np.load(m_0_file))
    sim.add(Demag())
    sim.add(Exchange(A))

    Hx = -24.6e-3 / mu0
    Hy = 4.3e-3 / mu0
    Hz = 0
    sim.add(Zeeman((Hx, Hy, Hz)))

    # The first zero-crossing of <m_x> is recorded whether or not we stop there,
    # so the full 2 ns trace (for plot_averages.py) can be produced while still
    # asserting the crossing time.
    #
    # The crossing is LINEARLY INTERPOLATED between the last sample with
    # <m_x> > 0 and the first with <m_x> <= 0. Reporting the raw sample time
    # instead (the legacy behaviour) biases the crossing late by up to one
    # sampling interval (1 ps), which is 12% of the derived tolerance below --
    # not negligible, so it is removed rather than absorbed. The interpolation
    # residual on 1 ps samples is < 0.01 ps (measured on the 10 ps-sampled
    # dynamics.ndt pre-flight trace, where linear vs cubic differ by 0.03 ps).
    crossing = {}
    previous = {}

    def check_if_crossed(sim):
        m_avg = np.array(sim.m_average)
        mx = m_avg[0]
        if mx <= 0:
            if "time" not in crossing:
                if previous:
                    # bracket [t_prev, t] with mx_prev > 0 >= mx
                    crossing["time"], crossing["m"] = interpolate_to_mx_zero(
                        previous["t"], previous["m"], sim.t, m_avg)
                else:
                    # first sample already past the crossing: nothing to
                    # interpolate against, report it as-is.
                    crossing["time"] = sim.t
                    crossing["m"] = m_avg
                print("The average m_x first crossed zero at t = {} "
                      "(interpolated), <m> = {}.".format(
                          crossing["time"], crossing["m"]))
                np.save(m_at_crossing_file, sim.m)
            # Return True -> "event done, keep running" (full trace);
            # return False -> stop the simulation at the crossing.
            return not stop_when_mx_eq_zero
        previous["t"], previous["m"] = sim.t, m_avg

    sim.schedule(check_if_crossed, every=1e-12)
    sim.schedule('save_averages', every=10e-12, at_end=True)
    sim.schedule('save_vtk', every=10e-12, at_end=True, overwrite=True)
    sim.run_until(2.0e-9)
    return sim.t, crossing.get("time"), crossing.get("m")


def reference_crossing():
    """First <m_x>=0 crossing of the checked-in Martinez reference trajectory.

    ``m_averages_ref_martinez.txt`` holds the muMAG standard problem #4 field-1
    reference of Martinez et al. as four whitespace-separated columns
    ``t[ns]  <m_x>  <m_y>  <m_z>``, 2500 rows on a uniform grid of pi/2500 ns
    (1.2566 ps) spanning 0.00126 .. 3.14159 ns.

    Returns ``(t_ref, my_ref, dmx_dt, dmy_dt)`` with ``t_ref`` in ns, the
    slopes in 1/ns, all evaluated at the linearly-interpolated crossing.

    NOTE (SR1 S3): the widely-quoted reference crossing time 0.13949 ns
    (http://www.ctcms.nist.gov/~rdm/std4/Torres.html) is NOT the interpolated
    crossing -- it is exactly grid point 111 of this file
    (111 * pi/2500 = 0.139487 ns), i.e. the first TABULATED sample whose
    <m_x> is already negative (<m_x> = -0.02007 there; the preceding sample
    at 0.13823 ns still has <m_x> = +0.00040). The interpolated crossing of
    the same reference trajectory is 0.138255 ns -- 1.23 ps EARLIER. The
    published figure therefore carries up to one sample interval of
    late-bias, which is 15% of the tolerance derived below, so this function
    interpolates rather than reusing the quoted number.
    """
    t, mx, my, _ = np.loadtxt(reference_file, unpack=True)
    i = int(np.flatnonzero(mx <= 0)[0])
    dt = t[i] - t[i - 1]
    t_ref, m_ref = interpolate_to_mx_zero(
        t[i - 1], (mx[i - 1], my[i - 1]), t[i], (mx[i], my[i]))
    dmx_dt = (mx[i] - mx[i - 1]) / dt
    dmy_dt = (my[i] - my[i - 1]) / dt
    return t_ref, m_ref[1], dmx_dt, dmy_dt


def derived_tolerances():
    """A-PRIORI tolerances for the quantitative muMAG anchor. See
    assert_matches_reference() for the derivation. Returns ``(tol_t_ns,
    tol_my)``."""
    _, _, dmx_dt, dmy_dt = reference_crossing()
    l_ex_nm = sqrt(2 * A / (mu0 * Ms ** 2)) * 1e9      # 5.6858 nm
    eps = 0.125 * (MESH_H_NM / l_ex_nm) ** 2           # 0.131421
    tol_t = eps / abs(dmx_dt)                          # 0.0080894 ns
    tol_my = eps * (1.0 + abs(dmy_dt) / abs(dmx_dt))   # 0.16339
    return tol_t, tol_my


def assert_matches_reference(t_cross_s, my_cross):
    """Quantitative muMAG anchor: crossing time and <m_y> at the crossing.

    TOLERANCE DERIVATION (SR1 S3) -- derived a priori, from the mesh and the
    reference trajectory alone, BEFORE any simulation output was compared.
    No number below was chosen to make an observed result pass.

    (1) Discretisation error scale eps in a component of <m>.
        The port meshes bar.geo with Gmsh OCC, not the legacy Netgen
        tetrahedraliser, so the two discretisations differ at the
        discretisation level (same geometry, same material, same maxh
        directive; different element layout). For P1 elements the pointwise
        interpolation error of a field varying on a length scale L over
        elements of diameter h is bounded by (h^2/8)*max|d2m/ds2|, and the
        sharpest micromagnetic structure varies on the exchange length
            l_ex = sqrt(2A/(mu0*Ms^2)) = 5.6858 nm,
        so max|d2m/ds2| <= 1/l_ex^2 (conservative: for a tanh wall of width
        l_ex the true maximum is 0.385/l_ex^2). With the measured
        representative element diameter h = 5.830 nm (see MESH_H_NM):
            eps = (1/8)*(h/l_ex)^2 = 0.125 * 1.0513 = 0.1314.
        The 1D-tight constant 1/8 is used rather than the crude
        multi-dimensional 1/2 because the reversal structure varies
        essentially along one direction (the 500 nm axis) while h is
        isotropic. Recorded for transparency: with 1/2 the time tolerance
        below would be 32.4 ps, only ~1.2x tighter than the coarse window --
        i.e. the choice of constant is what makes this anchor worth having.
        eps is an UPPER bound on the pointwise error and takes no credit for
        the cancellation that volume-averaging gives <m_x>; the anchor is
        therefore conservative by construction. (Corroboration, not an input:
        the t=0 s-state <m_x> differs from the reference's first sample by
        1.6e-3, ~80x below eps.)

    (2) Crossing-time sensitivity. A displacement eps of <m_x> near the
        crossing moves the crossing by eps / |d<m_x>/dt|. The slope is read
        NUMERICALLY from the Martinez reference at its own crossing
        (two-point difference across the bracketing samples):
            d<m_x>/dt = -16.246 /ns.
            tol_t = eps / 16.246 /ns = 0.0080894 ns = 8.09 ps.
        Sanity floor: the coarse window 0.10-0.18 ns admits +-40 ps about the
        crossing, so this is 4.9x tighter -- materially tighter, so the anchor
        adds real constraint rather than restating the window.

    (3) <m_y> at the crossing. Both trajectories are sampled on the same
        m_x = 0 section, so the comparison is of phase-space position, but a
        discretisation displacement eps in <m_x> still moves where the
        section is met, dragging <m_y> along by |d<m_y>/dt| * tol_t:
            d<m_y>/dt = -3.952 /ns,
            tol_my = eps * (1 + 3.952/16.246) = 0.1634.
        Against the reference value <m_y> = 0.7340 this excludes 92% of the
        a-priori range [-1, 1], where the current suite constrains <m_y> not
        at all.

    Reference values (recomputed at runtime from the checked-in file, quoted
    here for audit): t_ref = 0.138255 ns, <m_y>_ref = 0.733963.

    STATUS: pre-flighted against the 10 ps-sampled dynamics.ndt trace this
    port produced on 2026-07-28 (partial run, 0 .. 0.55 ns, which contains
    the crossing): interpolated t_cross = 0.13751 ns (|dt| = 0.75 ps, 11x
    inside tolerance) and <m_y> = 0.7299 (|d| = 0.0041, 40x inside; 0.0016
    with a cubic fit, the 10 ps sampling being the limiting factor there).
    FULL-RESOLUTION CONFIRMATION IS PENDING the post-performance-optimisation
    FULL-lane run -- this anchor has not yet been exercised end-to-end on a
    completed 2 ns full-resolution trajectory.
    """
    t_ref, my_ref, dmx_dt, dmy_dt = reference_crossing()
    tol_t, tol_my = derived_tolerances()
    t_cross = t_cross_s * 1e9  # s -> ns, the reference file's unit

    print("muMAG anchor: t_cross = {:.6f} ns vs reference {:.6f} ns "
          "(delta {:+.4f} ps, tolerance {:.4f} ps)".format(
              t_cross, t_ref, (t_cross - t_ref) * 1e3, tol_t * 1e3))
    print("muMAG anchor: <m_y> at crossing = {:.6f} vs reference {:.6f} "
          "(delta {:+.5f}, tolerance {:.5f})".format(
              my_cross, my_ref, my_cross - my_ref, tol_my))

    assert abs(t_cross - t_ref) < tol_t, (
        "first <m_x>=0 crossing at {:.6f} ns deviates from the Martinez "
        "reference crossing {:.6f} ns by {:+.4f} ps, outside the derived "
        "tolerance {:.4f} ps".format(
            t_cross, t_ref, (t_cross - t_ref) * 1e3, tol_t * 1e3))
    assert abs(my_cross - my_ref) < tol_my, (
        "<m_y> at the crossing is {:.6f}, deviating from the Martinez "
        "reference value {:.6f} by {:+.5f}, outside the derived tolerance "
        "{:.5f}".format(my_cross, my_ref, my_cross - my_ref, tol_my))


def test_std_prob_4_field_1(stop_when_mx_eq_zero=True):
    """muMAG standard problem #4, field 1.

    Always asserts the coarse physical switching window (0.10-0.18 ns). In
    FULL mode (FINMAG_EXAMPLE_FULL=1 -- the only mode in which the examples
    gate runs this script, see examples/test_examples.py SLOW_EXAMPLES) it
    additionally asserts the quantitative anchor against the published
    Martinez reference; the tolerance derivation lives in
    assert_matches_reference().
    """
    if not os.path.exists(m_0_file):
        print("Couldn't find initial magnetisation, creating one.")
        create_initial_s_state()

    print("Running simulation...")
    final_t, t_cross, m_cross = run_simulation(stop_when_mx_eq_zero)
    # When stop_when_mx_eq_zero is False the sim runs the full 2 ns, so the
    # crossing time (not the final time) is the quantity to compare.
    t_0 = t_cross if t_cross is not None else final_t

    t_ref_martinez = 0.13949e-9  # http://www.ctcms.nist.gov/~rdm/std4/Torres.html
    print("crossing time = {} s (Martinez reference {} s)".format(t_0, t_ref_martinez))
    # Coarse always-on guard (reduced mode asserts only this).
    assert 0.10e-9 < t_0 < 0.18e-9, \
        "average-m_x crossing time {} outside the physical window".format(t_0)

    if FULL:
        assert m_cross is not None, \
            "no <m_x>=0 crossing was recorded, cannot anchor against muMAG"
        assert_matches_reference(t_0, m_cross[1])
    else:
        print("Set FINMAG_EXAMPLE_FULL=1 for the quantitative muMAG anchor.")


if __name__ == "__main__":
    # stop_when_mx_eq_zero=False (legacy __main__): run the full 2 ns so the
    # complete <m>(t) trace is written for plot_averages.py; the crossing-time
    # assertion still runs against the recorded first crossing.
    test_std_prob_4_field_1(stop_when_mx_eq_zero=False)
    print("std_prob_4: average m_x crosses zero within the muMAG switching window.")
