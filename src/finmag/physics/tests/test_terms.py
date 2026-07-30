import pytest
import json
from os import path
try:  # master: import finmag.physics.equation as eqn
    import finmag.physics.equation as eqn
except ImportError:
    eqn = None  # not ported (D33): tests below xfail


@pytest.fixture
def terms_module():
    """
    Return whichever terms backend is active on this stack.

    The basic term-identity tests below should hold whether the current stack
    uses the live native backend or the Python fallback. [Codex GPT-5.4]
    """
    return eqn.get_terms_module()


def load_reference_data():
    """
    Load the checked-in native terms/equation reference payload.

    This keeps the Python fallback tied to backend-derived results even after
    the compiled container path is retired. [Codex GPT-5.4]
    """
    data_file = path.join(path.dirname(__file__), "equation_reference_data.json")
    with open(data_file, "r") as f:
        return json.load(f)


@pytest.fixture
def native_and_python_terms_modules():
    """
    Return both terms backends when the compiled hook still exists.

    These parity tests intentionally skip on stacks that no longer expose the
    legacy compiled extension entry point. [Codex GPT-5.4]
    """
    if not eqn.native_equation_module_available():
        pytest.skip("compiled equation backend is not available on this DOLFIN stack")
    return eqn.get_native_terms_module(), eqn.get_python_terms_module()


@pytest.mark.not_ported
@pytest.mark.xfail(reason="not ported: compiled terms backend (register M3)", strict=True)
def test_damping(terms_module):
    alpha, gamma = 1, 1
    mx, my, mz = 1, 0, 0
    Hx, Hy, Hz = 0, 1, 0
    dmx, dmy, dmz = terms_module.damping(alpha, gamma, mx, my, mz, Hx, Hy, Hz, 0, 0, 0)
    assert (dmx, dmy, dmz) == (0, 0.5, 0)


@pytest.mark.not_ported
@pytest.mark.xfail(reason="not ported: compiled terms backend (register M3)", strict=True)
def test_precession(terms_module):
    alpha, gamma = 1, 1
    mx, my, mz = 1, 0, 0
    Hx, Hy, Hz = 0, 1, 0
    dmx, dmy, dmz = terms_module.precession(alpha, gamma, mx, my, mz, Hx, Hy, Hz, 0, 0, 0)
    assert (dmx, dmy, dmz) == (0, 0, -0.5)


@pytest.mark.not_ported
@pytest.mark.xfail(reason="not ported: compiled terms backend (register M3)", strict=True)
def test_relaxation(terms_module):
    c = 1.0
    mx, my, mz = 2, 0, 0
    dmx, dmy, dmz = terms_module.relaxation(c, mx, my, mz, 0, 0, 0)
    assert (dmx, dmy, dmz) == (-6, 0, 0)


@pytest.mark.not_ported
@pytest.mark.xfail(reason="not ported: compiled terms backend (register M3)", strict=True)
def test_python_terms_match_native_terms(native_and_python_terms_modules):
    """
    Compare live native and Python terms outputs directly.

    This runs only while the compiled backend still exists and gives a direct
    parity signal in addition to the simpler formula-based tests above.
    [Codex GPT-5.4]
    """
    native_terms, python_terms = native_and_python_terms_modules
    damping_args = (1.2, 2.3, 0.6, -0.2, 0.7, -1.1, 0.4, 3.2, 0.1, -0.2, 0.3)
    precession_args = (0.7, 2.1, -0.5, 0.4, 0.3, 1.5, -0.6, 0.9, 0.2, 0.5, -0.4)
    relaxation_args = (1.7, 1.4, -0.8, 0.6, -0.2, 0.1, 0.3)

    assert native_terms.damping(*damping_args) == pytest.approx(
        python_terms.damping(*damping_args))
    assert native_terms.precession(*precession_args) == pytest.approx(
        python_terms.precession(*precession_args))
    assert native_terms.relaxation(*relaxation_args) == pytest.approx(
        python_terms.relaxation(*relaxation_args))


@pytest.mark.not_ported
@pytest.mark.xfail(reason="not ported: compiled terms backend (register M3)", strict=True)
def test_python_terms_match_checked_in_native_reference():
    """
    Compare Python terms outputs against stored native reference data.

    This keeps a backend-derived oracle available even after the live native
    container path disappears. [Codex GPT-5.4]
    """
    reference_data = load_reference_data()
    python_terms = eqn.get_python_terms_module()

    for name in ("damping", "precession", "relaxation"):
        case = reference_data["terms"][name]
        result = getattr(python_terms, name)(*case["args"])
        assert result == pytest.approx(case["result"])
