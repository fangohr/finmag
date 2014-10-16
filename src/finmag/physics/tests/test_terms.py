import pytest
from os import path
import dolfin as df


@pytest.fixture
def terms_module():
    MODULE_DIR = path.dirname(path.abspath(__file__))
    SOURCE_DIR = path.join(MODULE_DIR, "..", "native")

    with open(path.join(SOURCE_DIR, "terms.h"), "r") as header:
        code = header.read()

    extension_module = df.compile_extension_module(
        code=code,
        source_directory=SOURCE_DIR,
        sources=["terms.cpp"],
        # declare dm_x, dm_y and dm_z as input/output parameters
        # they will turn up in Python as return values
        additional_declarations="%apply double& INOUT { double& dm_x, double& dm_y, double& dm_z };",
        include_dirs=[SOURCE_DIR],)
    return extension_module


def test_damping(terms_module):
    alpha, gamma = 1, 1
    mx, my, mz = 1, 0, 0
    Hx, Hy, Hz = 0, 1, 0
    dmx, dmy, dmz = terms_module.damping(alpha, gamma, mx, my, mz, Hx, Hy, Hz, 0, 0, 0)
    assert (dmx, dmy, dmz) == (0, 0.5, 0)


def test_precession(terms_module):
    alpha, gamma = 1, 1
    mx, my, mz = 1, 0, 0
    Hx, Hy, Hz = 0, 1, 0
    dmx, dmy, dmz = terms_module.precession(alpha, gamma, mx, my, mz, Hx, Hy, Hz, 0, 0, 0)
    assert (dmx, dmy, dmz) == (0, 0, -0.5)


def test_relaxation(terms_module):
    c = 1.0
    mx, my, mz = 2, 0, 0
    dmx, dmy, dmz = terms_module.relaxation(c, mx, my, mz, 0, 0, 0)
    assert (dmx, dmy, dmz) == (-6, 0, 0)
