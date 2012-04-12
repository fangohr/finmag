import dolfin as df
import numpy as np

from finmag.sim.llg import LLG
from finmag.util.oommf import oommf_uniform_exchange, oommf_uniaxial_anisotropy

def compare_anisotropy(m_gen, K1, axis, dolfin_mesh, oommf_mesh, dims=3, name=""):
    finmag_anis_field, finmag = compute_finmag_anis(m_gen, K1, norm_axis(axis), dolfin_mesh)
    finmag_anis = finmag_to_oommf(finmag_anis_field, oommf_mesh, dims)
    oommf_anis = oommf_uniaxial_anisotropy(oommf_m0(m_gen, oommf_mesh), finmag.Ms, K1, axis).flat

    difference = np.abs(finmag_anis - oommf_anis)
    relative_difference = difference / np.sqrt(
        oommf_anis[0]**2 + oommf_anis[1]**2 + oommf_anis[2]**2)

    return dict(name=name, m0=finmag.m,
            mesh=dolfin_mesh, oommf_mesh=oommf_mesh,
            anis=finmag_anis, oommf_anis=oommf_anis,
            diff=difference, rel_diff=relative_difference)

def norm_axis(a):
    a = 1.0 * np.array(a)
    a /= np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
    return tuple(a) 

def compute_finmag_anis(m_gen, K1, axis, dolfin_mesh):
    coords = np.array(zip(* dolfin_mesh.coordinates()))
    m0 = m_gen(coords).flatten()

    llg = LLG(dolfin_mesh)
    llg.set_m(m0)
    llg.add_uniaxial_anisotropy(K1, df.Constant(axis))
    llg.setup(use_exchange=False)
    anis_field = df.Function(llg.V)
    anis_field.vector()[:] = llg._anisotropies[0].compute_field()
    return anis_field, llg

def compare_exchange(m_gen, dolfin_mesh, oommf_mesh, dims=3, name=""):
    finmag_exc_field, finmag = compute_finmag_exc(dolfin_mesh, m_gen)
    finmag_exc = finmag_to_oommf(finmag_exc_field, oommf_mesh, dims)
    oommf_exc = oommf_uniform_exchange(oommf_m0(m_gen, oommf_mesh), finmag.Ms, finmag.A).flat

    difference = np.abs(finmag_exc - oommf_exc)
    relative_difference = difference / np.sqrt(
        oommf_exc[0]**2 + oommf_exc[1]**2 + oommf_exc[2]**2)

    return dict(name=name, m0=finmag.m,
            mesh=dolfin_mesh, oommf_mesh=oommf_mesh,
            exc=finmag_exc, oommf_exc=oommf_exc,
            diff=difference, rel_diff=relative_difference)

def compute_finmag_exc(dolfin_mesh, m_gen):
    coords = np.array(zip(* dolfin_mesh.coordinates()))
    m0 = m_gen(coords).flatten()

    llg = LLG(dolfin_mesh)
    llg.set_m(m0)
    llg.setup()
    finmag_exc_field = df.Function(llg.V)
    finmag_exc_field.vector()[:] = llg.exchange.compute_field()
    return finmag_exc_field, llg

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
