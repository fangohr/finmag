import pytest
import numpy as np
import dolfin as df
import json
from time import perf_counter
from os import path
import finmag.physics.equation as eqn
from finmag.physics.equation import Equation


@pytest.fixture
def setup():
    mesh = df.UnitIntervalMesh(3)
    V = df.FunctionSpace(mesh, "CG", 1)
    alpha = df.Function(V)
    alpha.assign(df.Constant(1))
    W = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    m = df.Function(W)
    m.assign(df.Constant((0.6, 0.8, 0)))
    H = df.Function(W)
    H.assign(df.Constant((1, 2, 3)))
    dmdt = df.Function(W)
    return mesh, V, alpha, W, m, H, dmdt


def setup_for_debugging():
    """
    Sets up equation for greater convenience during interactive debugging.

    """
    mesh, V, alpha, W, m, H, dmdt = setup()
    equation = Equation(m.vector(), H.vector(), dmdt.vector())
    equation.set_alpha(alpha.vector())
    equation.set_gamma(1.0)
    equation.solve()
    return {'mesh': mesh,
            'V': V, 'alpha': alpha,
            'W': W, 'm': m, 'H': H, 'dmdt': dmdt,
            'equation': equation}


def same(v, w, TOL=1e-14):
    """
    Returns True if the vectors `v` and `w` have the same entries.

    """
    diff = v.get_local() - w.get_local()
    print("v = {}\nw = {}\ndiff = {}".format(v.get_local(), w.get_local(), diff))
    return np.sum(np.abs(diff)) < TOL


def build_equation(equation_module, alpha, W, m, H):
    """
    Build an Equation backend from shared field state.

    This keeps the native-backend parity checks and the checked-in reference
    checks aligned on identical finite-element inputs, so any mismatch points
    to backend behaviour rather than setup drift. [Codex GPT-5.4]
    """
    dmdt = df.Function(W)
    equation = equation_module.Equation(m.vector(), H.vector(), dmdt.vector())
    equation.set_alpha(alpha.vector())
    equation.set_gamma(1.0)
    return equation, dmdt


def load_reference_data():
    """
    Load checked-in legacy-native reference outputs.

    The compiled backend will eventually disappear from active environments,
    so these stored native results let the Python fallback keep a direct
    numerical link to the implementation it replaces. [Codex GPT-5.4]
    """
    data_file = path.join(path.dirname(__file__), "equation_reference_data.json")
    with open(data_file, "r") as f:
        return json.load(f)


@pytest.fixture
def native_and_python_equation_modules():
    """
    Return both Equation backends when the native hook still exists.

    These parity tests intentionally skip on pixi/FEniCS-2019, where the old
    compiled extension entry point no longer exists. [Codex GPT-5.4]
    """
    if not eqn.native_equation_module_available():
        pytest.skip("compiled equation backend is not available on this DOLFIN stack")
    return eqn.get_native_equation_module(True), eqn.get_python_equation_module()


def test_new_equation(setup):
    mesh, V, alpha, W, m, H, dmdt = setup
    equation = Equation(m.vector(), H.vector(), dmdt.vector())


def test_new_equation_wrong_size(setup):
    mesh, V, alpha, W, m, H, dmdt = setup
    W = df.VectorFunctionSpace(mesh, "CG", 2, dim=3)  # W like Wrong
    H_W = df.Function(W)
    with pytest.raises(Exception):
        equation = Equation(m.vector(), H_W.vector(), dmdt.vector())


def test_regression_vector_wrong_state(setup):
    mesh, V, alpha, W, m, H, dmdt = setup
    equation = Equation(m.vector(), H.vector(), dmdt.vector())
    equation.set_alpha(alpha.vector())
    equation.set_gamma(1.0)
    equation.solve()
    # the following operation would fail with PETSc error code 73
    # saying the vector is in wrong state. An "apply" call in the C++
    # code fixes this.
    operation = dmdt.vector() - m.vector()


def test_alpha_not_set(setup):
    mesh, V, alpha, W, m, H, dmdt = setup
    equation = Equation(m.vector(), H.vector(), dmdt.vector())
    assert equation.get_alpha() is None  # doesn't crash
    with pytest.raises(RuntimeError):
        equation.solve()


def test_alpha_keeps_track_of_change(setup):
    mesh, V, alpha, W, m, H, dmdt = setup
    equation = Equation(m.vector(), H.vector(), dmdt.vector())
    equation.set_alpha(alpha.vector())
    assert same(alpha.vector(), equation.get_alpha())
    # since alpha and Equation::alpha are fundamentally the same object
    # changing one should change the other, which is what we test next
    alpha.assign(df.Constant(2))
    assert same(alpha.vector(), equation.get_alpha())


def test_solve(setup):
    mesh, V, alpha, W, m, H, dmdt = setup
    equation = Equation(m.vector(), H.vector(), dmdt.vector())
    equation.set_alpha(alpha.vector())
    equation.set_gamma(1.0)
    equation.solve()

    dmdt_expected = df.Function(W)
    #dmdt_expected.assign(df.Constant((0.0, 0.5, -0.5)))
    dmdt_expected.assign(df.Constant((-1.36, 1.02, 1.3)))
    assert same(dmdt.vector(), dmdt_expected.vector())


def test_pinning(setup):
    mesh, V, alpha, W, m, H, dmdt = setup
    equation = Equation(m.vector(), H.vector(), dmdt.vector())
    equation.set_alpha(alpha.vector())
    equation.set_gamma(1.0)
    pins = df.Function(V)
    pins.vector()[0] = 1  # pin first node, but this could be done using an expression
    equation.set_pinned_nodes(pins.vector())
    equation.solve()
    dmdt_values = dmdt.vector().get_local().reshape(3, -1)
    dmdt_node0 = dmdt_values[:, 0]
    dmdt_node_others = dmdt_values[:, 1:]
    # check that first node is pinned, i.e. dmdt = 0 there
    assert np.all(dmdt_node0 == np.array((0, 0, 0)))
    # check that we don't accidentally set the whole dmdt array to zero
    assert not np.all(dmdt_node_others == 0)


def test_slonczewski(setup):
    mesh, V, alpha, W, m, H, dmdt = setup
    equation = Equation(m.vector(), H.vector(), dmdt.vector())
    equation.set_alpha(alpha.vector())
    equation.set_gamma(1.0)

    Ms = df.Function(V)
    Ms.assign(df.Constant(1))
    J = df.Function(V)
    J.assign(df.Constant(1))
    equation.slonczewski(5e-9, 0.4, np.array((1.0, 0.0, 0.0)), 1, 0)
    assert equation.slonczewski_status() is False  # missing J, Ms
    equation.set_saturation_magnetisation(Ms.vector())
    equation.set_current_density(J.vector())
    assert equation.slonczewski_status() is True
    equation.solve()


def test_python_equation_matches_checked_in_reference_solve(setup):
    """
    Check the Python fallback against stored native solve output.

    This test keeps working after the container-native backend disappears,
    because it compares against checked-in legacy-native results rather than a
    live compiled module. [Codex GPT-5.4]
    """
    mesh, V, alpha, W, m, H, dmdt = setup
    reference_data = load_reference_data()
    python_equation, python_dmdt = build_equation(
        eqn.get_python_equation_module(), alpha, W, m, H)
    python_equation.solve()

    assert np.allclose(
        python_dmdt.vector().get_local(),
        reference_data["equation"]["solve"],
        atol=1e-13,
        rtol=0.0)


def test_python_equation_matches_checked_in_reference_pinning(setup):
    """
    Check the Python fallback pinning branch against stored native output.

    Pinning is easy to get subtly wrong in a node-wise Python loop, so keep a
    backend-derived oracle even on stacks without the compiled module.
    [Codex GPT-5.4]
    """
    mesh, V, alpha, W, m, H, dmdt = setup
    reference_data = load_reference_data()
    python_equation, python_dmdt = build_equation(
        eqn.get_python_equation_module(), alpha, W, m, H)
    pins = df.Function(V)
    pins.vector()[0] = 1
    python_equation.set_pinned_nodes(pins.vector())
    python_equation.solve()

    assert np.allclose(
        python_dmdt.vector().get_local(),
        reference_data["equation"]["pinning"],
        atol=1e-13,
        rtol=0.0)


def test_python_equation_matches_checked_in_reference_slonczewski(setup):
    """
    Check the Python Slonczewski branch against stored native output.

    This guards the fallback spin-torque implementation after the legacy
    container path is no longer available. [Codex GPT-5.4]
    """
    mesh, V, alpha, W, m, H, dmdt = setup
    reference_data = load_reference_data()
    python_equation, python_dmdt = build_equation(
        eqn.get_python_equation_module(), alpha, W, m, H)
    Ms = df.Function(V)
    Ms.assign(df.Constant(1))
    J = df.Function(V)
    J.assign(df.Constant(1))
    python_equation.slonczewski(5e-9, 0.4, np.array((1.0, 0.0, 0.0)), 1, 0)
    python_equation.set_saturation_magnetisation(Ms.vector())
    python_equation.set_current_density(J.vector())
    python_equation.solve()

    assert np.allclose(
        python_dmdt.vector().get_local(),
        reference_data["equation"]["slonczewski"],
        atol=1e-13,
        rtol=0.0)


def test_python_equation_matches_native_solve(setup, native_and_python_equation_modules):
    """
    Compare live native and Python Equation.solve() outputs directly.

    This test only runs while the compiled backend still exists and gives a
    stronger signal than checking the fallback only against handwritten values.
    [Codex GPT-5.4]
    """
    mesh, V, alpha, W, m, H, dmdt = setup
    native_module, python_module = native_and_python_equation_modules
    native_equation, native_dmdt = build_equation(native_module, alpha, W, m, H)
    python_equation, python_dmdt = build_equation(python_module, alpha, W, m, H)

    native_equation.solve()
    python_equation.solve()

    assert np.allclose(
        native_dmdt.vector().get_local(),
        python_dmdt.vector().get_local(),
        atol=1e-13,
        rtol=0.0)


def test_python_equation_matches_native_pinning(setup, native_and_python_equation_modules):
    """
    Compare live native and Python pinning outputs directly.

    Pinning has its own control-flow branch, so it is kept as a separate
    native-parity check while the compiled backend still exists.
    [Codex GPT-5.4]
    """
    mesh, V, alpha, W, m, H, dmdt = setup
    native_module, python_module = native_and_python_equation_modules
    native_equation, native_dmdt = build_equation(native_module, alpha, W, m, H)
    python_equation, python_dmdt = build_equation(python_module, alpha, W, m, H)
    pins = df.Function(V)
    pins.vector()[0] = 1

    native_equation.set_pinned_nodes(pins.vector())
    python_equation.set_pinned_nodes(pins.vector())
    native_equation.solve()
    python_equation.solve()

    assert np.allclose(
        native_dmdt.vector().get_local(),
        python_dmdt.vector().get_local(),
        atol=1e-13,
        rtol=0.0)


def test_python_equation_jtimes_matches_finite_difference(setup):
    """
    Check Python jtimes against its finite-difference contract.

    The compiled SWIG binding does not expose the native jtimes output in a
    NumPy-friendly way, so this test validates the fallback against the
    backend-independent mathematical contract instead. [Codex GPT-5.4]
    """
    mesh, V, alpha, W, m, H, dmdt = setup
    python_equation, python_dmdt = build_equation(
        eqn.get_python_equation_module(), alpha, W, m, H)
    size = m.vector().local_size()
    mp = np.linspace(0.1, 0.1 * size, size, dtype=float)
    Hp = np.linspace(-0.2, 0.05 * (size - 4), size, dtype=float)
    python_jtimes = np.zeros(size, dtype=float)
    eps = 1e-8

    python_equation.sundials_jtimes_serial(mp, Hp, python_jtimes)

    vec_m_plus = m.vector().copy()
    vec_H_plus = H.vector().copy()
    vec_dmdt_plus = python_dmdt.vector().copy()
    vec_m_minus = m.vector().copy()
    vec_H_minus = H.vector().copy()
    vec_dmdt_minus = python_dmdt.vector().copy()
    vec_m_plus.set_local(m.vector().get_local() + eps * mp)
    vec_H_plus.set_local(H.vector().get_local() + eps * Hp)
    vec_m_minus.set_local(m.vector().get_local() - eps * mp)
    vec_H_minus.set_local(H.vector().get_local() - eps * Hp)
    vec_m_plus.apply("")
    vec_H_plus.apply("")
    vec_m_minus.apply("")
    vec_H_minus.apply("")

    python_equation.solve_with(vec_m_plus, vec_H_plus, vec_dmdt_plus)
    python_equation.solve_with(vec_m_minus, vec_H_minus, vec_dmdt_minus)
    finite_difference = (vec_dmdt_plus.get_local() - vec_dmdt_minus.get_local()) / (2 * eps)

    assert np.allclose(python_jtimes, finite_difference, atol=1e-9, rtol=0.0)


def test_native_and_python_equation_benchmark_report(native_and_python_equation_modules):
    """
    Report native-vs-Python solve timing without gating on a hard threshold.

    The goal is to keep an approximate performance ratio in CI logs while the
    compiled backend still exists, without turning that signal into a flaky
    micro-benchmark assertion. [Codex GPT-5.4]
    """
    native_module, python_module = native_and_python_equation_modules
    mesh = df.UnitIntervalMesh(199)
    V = df.FunctionSpace(mesh, "CG", 1)
    alpha = df.Function(V)
    alpha.assign(df.Constant(1))
    W = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    m = df.Function(W)
    m.assign(df.Constant((0.6, 0.8, 0)))
    H = df.Function(W)
    H.assign(df.Constant((1, 2, 3)))
    native_equation, native_dmdt = build_equation(native_module, alpha, W, m, H)
    python_equation, python_dmdt = build_equation(python_module, alpha, W, m, H)

    native_equation.solve()
    python_equation.solve()

    repeats = 200
    start = perf_counter()
    for _ in range(repeats):
        native_equation.solve()
    native_seconds = perf_counter() - start

    start = perf_counter()
    for _ in range(repeats):
        python_equation.solve()
    python_seconds = perf_counter() - start

    assert np.allclose(
        native_dmdt.vector().get_local(),
        python_dmdt.vector().get_local(),
        atol=1e-13,
        rtol=0.0)
    # Record relative cost without freezing an exact threshold into the gate:
    # the value is meant to be inspected in CI logs, not tuned as a flaky
    # performance assertion. [Codex GPT-5.4]
    print("equation backend benchmark: native={:.6f}s python={:.6f}s ratio={:.2f}x".format(
        native_seconds, python_seconds, python_seconds / native_seconds))
    assert native_seconds > 0.0
    assert python_seconds > 0.0
