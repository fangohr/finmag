import dolfin as df
import numpy as np
from finmag.energies import UniaxialAnisotropy, Exchange
from finmag.util.oommf import oommf_uniform_exchange, oommf_uniaxial_anisotropy

def compare_anisotropy(m_gen, Ms, K1, axis, dolfin_mesh, oommf_mesh, dims=3, name=""):
    finmag_anis_field = compute_finmag_anis(m_gen, Ms, K1, norm_axis(axis), dolfin_mesh)
    finmag_anis = finmag_to_oommf(finmag_anis_field, oommf_mesh, dims)
    oommf_anis = oommf_uniaxial_anisotropy(oommf_m0(m_gen, oommf_mesh), Ms, K1, axis).flat

    difference = np.abs(finmag_anis - oommf_anis)
    relative_difference = difference / np.sqrt(
        oommf_anis[0]**2 + oommf_anis[1]**2 + oommf_anis[2]**2)

    return dict(name=name,
            mesh=dolfin_mesh, oommf_mesh=oommf_mesh,
            anis=finmag_anis, oommf_anis=oommf_anis,
            diff=difference, rel_diff=relative_difference)

def norm_axis(a):
    a = 1.0 * np.array(a)
    a /= np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
    return tuple(a) 

def compute_finmag_anis(m_gen, Ms, K1, axis, dolfin_mesh):
    S3 = df.VectorFunctionSpace(dolfin_mesh, "Lagrange", 1, dim=3)
    coords = np.array(zip(* dolfin_mesh.coordinates()))
    m0 = m_gen(coords).flatten()
    m = df.Function(S3)
    m.vector()[:] = m0

    anis = UniaxialAnisotropy(K1, axis)
    anis.setup(S3, m, Ms)

    anis_field = df.Function(S3)
    anis_field.vector()[:] = anis.compute_field()
    return anis_field

def compare_exchange(m_gen, Ms, A, dolfin_mesh, oommf_mesh, dims=3, name=""):
    finmag_exc_field = compute_finmag_exc(dolfin_mesh, m_gen, Ms, A)
    finmag_exc = finmag_to_oommf(finmag_exc_field, oommf_mesh, dims)
    oommf_exc = oommf_uniform_exchange(oommf_m0(m_gen, oommf_mesh), Ms, A).flat

    difference = np.abs(finmag_exc - oommf_exc)
    relative_difference = difference / np.sqrt(
        oommf_exc[0]**2 + oommf_exc[1]**2 + oommf_exc[2]**2)

    return dict(name=name,
            mesh=dolfin_mesh, oommf_mesh=oommf_mesh,
            exc=finmag_exc, oommf_exc=oommf_exc,
            diff=difference, rel_diff=relative_difference)

def compute_finmag_exc(dolfin_mesh, m_gen, Ms, A):
    S3 = df.VectorFunctionSpace(dolfin_mesh, "Lagrange", 1, dim=3)
    coords = np.array(zip(* dolfin_mesh.coordinates()))
    m0 = m_gen(coords).flatten()
    m = df.Function(S3)
    m.vector()[:] = m0

    exchange = Exchange(A)
    exchange.setup(S3, m, Ms)

    finmag_exc_field = df.Function(S3)
    finmag_exc_field.vector()[:] = exchange.compute_field()
    return finmag_exc_field

def oommf_m0(m_gen, oommf_mesh):
    coords = np.array(zip(* oommf_mesh.iter_coords()))
    m0 = oommf_mesh.new_field(3)
    m0.flat = m_gen(coords)
    m0.flat /= np.sqrt(
            m0.flat[0]**2 +m0.flat[1]**2 + m0.flat[2]**2)
    return m0 

def finmag_to_oommf(f, oommf_mesh, dims=1):
    """
    Given a dolfin.Function f and a mesh oommf_mesh as defined in
    finmag.util.oommf.mesh, it will probe the values of f at the coordinates
    of oommf_mesh and return the resulting, oommf_compatible mesh_field.

    """
    f_for_oommf = oommf_mesh.new_field(3)
    for i, (x, y, z) in enumerate(oommf_mesh.iter_coords()):
        if dims == 1:
            f_x, f_y, f_z = f(x)
        else:
            f_x, f_y, f_z = f(x, y, z)
        f_for_oommf.flat[0,i] = f_x
        f_for_oommf.flat[1,i] = f_y
        f_for_oommf.flat[2,i] = f_z
    return f_for_oommf.flat
