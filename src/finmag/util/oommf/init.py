from oommf_calculator import calculate_oommf_fields
import numpy as np
from mesh import MeshField

def mesh_spec(mesh):
    return """
Specify Oxs_BoxAtlas:atlas {
  xrange {%25.18e %25.18e}
  yrange {%25.18e %25.18e}
  zrange {%25.18e %25.18e}
}

Specify Oxs_RectangularMesh:mesh {
  cellsize {%25.18e %25.18e %25.18e}
  atlas Oxs_BoxAtlas:atlas
}
""" % (
    mesh.origin[0], mesh.endpoint[0],
    mesh.origin[1], mesh.endpoint[1],
    mesh.origin[2], mesh.endpoint[2],
    mesh.cell_size[0], mesh.cell_size[1], mesh.cell_size[2],
    )

def oommf_demag(s0, Ms):
    assert type(s0) is MeshField and s0.dims == (3,)

    res = calculate_oommf_fields("demag", s0, Ms, mesh_spec(s0.mesh) + "\nSpecify Oxs_Demag {}", fields=["Oxs_Demag::Field", "Oxs_TimeDriver::Spin"])
    demag_field = res['Oxs_Demag-Field']
    s_field = res['Oxs_TimeDriver-Spin']

    assert demag_field.dims == (3,)
    if not (np.max(np.abs(s_field.flat - s0.flat)) < 1e-14):
        print s_field.flat
        print s0.flat
    assert np.max(np.abs(s_field.flat - s0.flat)) < 1e-14

    return demag_field

def oommf_uniform_exchange(s0, Ms, A):
    assert type(s0) is MeshField and s0.dims == (3,)

    res = calculate_oommf_fields("uniform_exchange", s0, Ms, mesh_spec(s0.mesh) + "\nSpecify Oxs_UniformExchange { A %25.15e }" % A,
                                 fields=["Oxs_UniformExchange::Field", "Oxs_TimeDriver::Spin"])
    exchange_field = res['Oxs_UniformExchange-Field']
    s_field = res['Oxs_TimeDriver-Spin']

    assert exchange_field.dims == (3,)
    if not (np.max(np.abs(s_field.flat - s0.flat)) < 1e-14):
        print s_field.flat
        print s0.flat
    assert np.max(np.abs(s_field.flat - s0.flat)) < 1e-14

    return exchange_field

def oommf_uniaxial_anisotropy(m0, Ms, K1, axis):
    assert type(m0) is MeshField and m0.dims == (3,)

    res = calculate_oommf_fields("uniaxial_anisotropy", m0, Ms, mesh_spec(m0.mesh) + "\nSpecify Oxs_UniaxialAnisotropy { K1 %25.15e axis { %25.15e %25.15e %25.15e } }" % (K1, axis[0], axis[1], axis[2]),
                                 fields=["Oxs_UniaxialAnisotropy::Field", "Oxs_TimeDriver::Spin"])
    uniaxial_anisotropy_field = res['Oxs_UniaxialAnisotropy-Field']
    m_field = res['Oxs_TimeDriver-Spin']

    assert uniaxial_anisotropy_field.dims == (3,)
    if not (np.max(np.abs(m_field.flat - m0.flat)) < 1e-14):
        print m_field.flat
        print m0.flat
    assert np.max(np.abs(m_field.flat - m0.flat)) < 1e-14

    return uniaxial_anisotropy_field

def oommf_cubic_anisotropy(m0, Ms, u1, u2, K1, K2=0, K3=0):
    assert type(m0) is MeshField and m0.dims == (3,)

    res = calculate_oommf_fields("cubic_anisotropy", m0, Ms, mesh_spec(m0.mesh) + """\nSpecify Southampton_CubicAnisotropy8 { 
                                K1 %25.15e K2 %25.15e K3 %25.15e axis1 { %25.15e %25.15e %25.15e } 
                                axis2 { %25.15e %25.15e %25.15e } }""" % (K1, K2, K3, u1[0], u1[1], u1[2], u2[0], u2[1], u2[2]),
                                 fields=["Southampton_CubicAnisotropy8::Field", "Oxs_TimeDriver::Spin"])
    cubic_anisotropy_field = res['Southampton_CubicAnisotropy8-Field']
    m_field = res['Oxs_TimeDriver-Spin']

    assert cubic_anisotropy_field.dims == (3,)
    if not (np.max(np.abs(m_field.flat - m0.flat)) < 1e-14):
        print m_field.flat
        print m0.flat
    assert np.max(np.abs(m_field.flat - m0.flat)) < 1e-14

    return cubic_anisotropy_field

def oommf_fixed_zeeman(s0, Ms, H):
    assert type(s0) is MeshField and s0.dims == (3,)

    res = calculate_oommf_fields("fixed_zeeman", s0, Ms, mesh_spec(s0.mesh) + "\nSpecify Oxs_FixedZeeman { field {%25.16e %25.16e %25.16e} }" % (H[0], H[1], H[2]),
                                 fields=["Oxs_FixedZeeman::Field", "Oxs_TimeDriver::Spin"])
    field = res['Oxs_FixedZeeman-Field']
    s_field = res['Oxs_TimeDriver-Spin']

    assert field.dims == (3,)
    if not (np.max(np.abs(s_field.flat - s0.flat)) < 1e-14):
        print s_field.flat
        print s0.flat
    assert np.max(np.abs(s_field.flat - s0.flat)) < 1e-14

    return field

def oommf_dmdt(s0, Ms, A, H, alpha, gamma_G):
    assert type(s0) is MeshField and s0.dims == (3,)

    # disable everything besides the external field for better comparison.
    res = calculate_oommf_fields("dmdt", s0, Ms, mesh_spec(s0.mesh) +
        "\nSpecify Oxs_FixedZeeman { field {%25.16e %25.16e %25.16e} }" % (H[0], H[1], H[2]),
        alpha=alpha, gamma_G=gamma_G,
        fields=["Oxs_RungeKuttaEvolve:evolver:dm/dt", "Oxs_TimeDriver::Spin"])
    field = res['Oxs_RungeKuttaEvolve-evolver-dm_dt']
    s_field = res['Oxs_TimeDriver-Spin']

    assert field.dims == (3,)
    if not (np.max(np.abs(s_field.flat - s0.flat)) < 1e-14):
        print s_field.flat
        print s0.flat
    assert np.max(np.abs(s_field.flat - s0.flat)) < 1e-14

    return field
