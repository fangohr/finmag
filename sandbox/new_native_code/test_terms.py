import os
import dolfin as df

with open("native/terms.h", "r") as header:
    code = header.read()

terms_module = df.compile_extension_module(
        code=code,
        source_directory="native",
        sources=["terms.cpp"],
        # declare dm_x, dm_y and dm_z as input/output parameters
        # they will turn up in Python as return values
        additional_declarations="%apply double& INOUT { double& dm_x, double& dm_y, double& dm_z };",
        include_dirs=[".", os.path.abspath("native")],)

def test_damping():
    alpha, gamma = 1, 1
    mx, my, mz = 1, 0, 0
    Hx, Hy, Hz = 0, 1, 0
    dmx, dmy, dmz = terms_module.damping(alpha, gamma, mx, my, mz, Hx, Hy, Hz, 0, 0, 0)
    assert (dmx, dmy, dmz) == (0, 0.5, 0)


def test_precession():
    alpha, gamma = 1, 1
    mx, my, mz = 1, 0, 0
    Hx, Hy, Hz = 0, 1, 0
    dmx, dmy, dmz = terms_module.precession(alpha, gamma, mx, my, mz, Hx, Hy, Hz, 0, 0, 0)
    assert (dmx, dmy, dmz) == (0, 0, -0.5)


def test_relaxation():
    c = 1.0
    mx, my, mz = 2, 0, 0
    dmx, dmy, dmz = terms_module.relaxation(c, mx, my, mz, 0, 0, 0)
    assert (dmx, dmy, dmz) == (-6, 0, 0)



