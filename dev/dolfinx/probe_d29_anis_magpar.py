"""D29 probe: is the ~8% anisotropy-vs-Magpar residual mesh drift or a defect?

Register row D29 records that ``src/finmag/tests/comparison/anisotropy/
test_anis_magpar.py`` loosened master's ``REL_TOLERANCE`` from ``5e-7`` to
``8e-2`` (measured max ~5.1e-2), with the maximum arising at a node whose
coordinate coincides EXACTLY with a finmag vertex -- while the sibling exchange
comparison (``tests/comparison/exchange/test_exchange_compare_magpar.py``)
holds ``9e-8`` at exact node coincidence. That asymmetry left "mesh drift" vs
"masked anisotropy-field defect" unresolved.

This script settles it numerically. It never imports or modifies
``src/finmag/energies``'s internals beyond calling the public
``UniaxialAnisotropy`` API; the box-projection reference below is an
INDEPENDENT NumPy reimplementation (element-wise P1 mass/volume integrals), so
agreement with the shipped field is genuine corroboration, not a tautology.

Physics being tested
--------------------
Both finmag and Magpar compute the nodal anisotropy field by the *box method*
(lumped/vertex-volume projection of the variational derivative). For uniaxial
anisotropy with easy axis ``a`` and constant ``Ms``:

    E = int K1 (1 - (a.m_h)^2) dV
    H_i = -(1/(mu0 Ms V_i)) dE/dm_i
        = (2 K1/(mu0 Ms)) * a * [ int phi_i (a.m_h) dV / int phi_i dV ]

The bracket is a PATCH AVERAGE of the piecewise-linear interpolant of ``a.m``
over the elements touching node i -- NOT the pointwise value ``(a.m)(x_i)``.
For a P1 patch,

    int phi_i u_h dV / int phi_i dV  =  u(x_i) + grad(u).(c_i - x_i) + O(h^2)

where ``c_i = int phi_i x dV / int phi_i dV`` is the phi-weighted patch
centroid. On an unstructured mesh ``c_i != x_i``; the offset is O(h) and
depends ONLY on the local element geometry. Two different tetrahedralisations
of the same body therefore give DIFFERENT nodal box fields at the very same
coordinate -- by O(h * |grad m|), which for this bar (h_avg ~ 2.2 nm, |grad
m_x| ~ 0.05 / nm relative to |m_x|max ~ 0.48) is percent-level. That is the
size of the disputed residual.

The script therefore checks, in order:

  A. finmag's shipped H == independent box-projection on finmag's OWN mesh?
  B. finmag's H vs the POINTWISE analytic H (both codes must deviate; the box
     field is a patch average, so pointwise agreement is not expected).
  C. Magpar's stored Hani == the same independent box projection evaluated on
     MAGPAR's OWN mesh (from its .femsh connectivity and stored M)?
  D. Magpar's stored Hani vs the pointwise analytic H (same order as B?).
  E. At coincident nodes: is the observed finmag-vs-Magpar residual PREDICTED
     by the patch-centroid-offset difference grad(m_x).(c_fin - c_mag)?
  F. Controls on the m-dependence: with UNIFORM m (grad m = 0) the patch
     average is exact on any mesh, so the residual must collapse to round-off;
     with the real m it must not.

Run:  pixi run -e dolfinx python dev/dolfinx/probe_d29_anis_magpar.py
"""

import os
import sys

import numpy as np
from scipy.spatial import cKDTree

import dolfinx.fem as fem

from finmag.field import Field
from finmag.energies import UniaxialAnisotropy
from finmag.util.meshes import from_geofile
from finmag.util import magpar_io

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANIS_DIR = os.path.join(
    REPO, "src", "finmag", "tests", "comparison", "anisotropy")

# --- test parameters, verbatim from tests/comparison/anisotropy ------------
Ms = 0.86e6
K1 = 520e3
u1 = np.array([1.0, 0.0, 0.0])
x1 = y1 = z1 = 20.0
mu0 = 4e-7 * np.pi
PREF = 2 * K1 / (mu0 * Ms)          # 2 K1 / (mu0 Ms), the analytic prefactor
COINC_TOL = 1e-6


def m_gen(r):
    """Analytic unit magnetisation pattern (anisotropy/conftest.py)."""
    x = np.maximum(np.minimum(r[0] / x1, 1.0), 0.0)
    y = np.maximum(np.minimum(r[1] / y1, 1.0), 0.0)
    z = np.maximum(np.minimum(r[2] / z1, 1.0), 0.0)
    mx = (2 - y) * (2 * x - 1) / 4
    mz = (2 - y) * (2 * z - 1) / 4
    my = np.sqrt(1 - mx ** 2 - mz ** 2)
    return np.array([mx, my, mz])


def grad_mx(r):
    """Analytic grad of m_x = (2 - y/y1)(2 x/x1 - 1)/4 (per mesh unit, nm)."""
    y = np.clip(r[..., 1] / y1, 0.0, 1.0)
    x = np.clip(r[..., 0] / x1, 0.0, 1.0)
    gx = (2 - y) * 2 / (4 * x1)
    gy = -(2 * x - 1) / (4 * y1)
    gz = np.zeros_like(gx)
    return np.column_stack((gx, gy, gz))


# --- independent P1 box-projection reference (pure NumPy) -------------------
def _tet_volumes(nodes, cells):
    p = nodes[cells]                                   # (ncell, 4, 3)
    d = p[:, 1:, :] - p[:, :1, :]
    return np.abs(np.linalg.det(d)) / 6.0


def box_average(nodes, cells, u):
    """Lumped P1 patch average  int phi_i u_h dV / int phi_i dV.

    Uses the exact P1 mass matrix on a tetrahedron
    (int phi_i phi_j = V/20, i != j;  int phi_i^2 = V/10) and
    int phi_i = V/4.
    """
    vol = _tet_volumes(nodes, cells)
    n = nodes.shape[0]
    num = np.zeros(n)
    den = np.zeros(n)
    usum = u[cells].sum(axis=1)                        # (ncell,)
    for k in range(4):
        idx = cells[:, k]
        np.add.at(num, idx, vol * (usum + u[idx]) / 20.0)
        np.add.at(den, idx, vol / 4.0)
    return num / den, den


def patch_centroid(nodes, cells):
    """phi-weighted patch centroid c_i = int phi_i x dV / int phi_i dV."""
    c = np.column_stack([box_average(nodes, cells, nodes[:, d])[0]
                         for d in range(3)])
    return c


# --- finmag side -----------------------------------------------------------
def setup_bar_anis(m_callable=m_gen):
    """Verbatim copy of the test's ``_setup_bar_anis`` (also returns m)."""
    mesh = from_geofile(os.path.join(ANIS_DIR, "bar.geo"), save_result=False)
    S3 = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    CG = fem.functionspace(mesh, ("Lagrange", 1))
    m = Field(S3)
    m.set(m_callable)
    anis = UniaxialAnisotropy(K1, tuple(u1), K2=0)
    anis.setup(m, Field(CG, Ms), unit_length=1e-9)
    H = Field(S3)
    H.set_with_numpy_array_debug(anis.compute_field())
    return mesh, m, H


def finmag_cells(mesh):
    dofmap = mesh.geometry.dofmap
    return np.asarray(dofmap).reshape((-1, 4))


def stats(name, a):
    print("    {:<34s} max={:.3e}  mean={:.3e}".format(
        name, np.max(a), np.mean(a)))


def main():
    print("=" * 78)
    print("D29 probe -- anisotropy vs Magpar: mesh drift or field defect?")
    print("=" * 78)

    # ---------------------------------------------------------------- setup
    mesh, m, H = setup_bar_anis()
    fcoords, fH = H.coords_and_values()
    _, fm = m.coords_and_values()
    fcells = finmag_cells(mesh)
    fnodes = mesh.geometry.x[:fcoords.shape[0], :3]
    print("\n[0] MESHES")
    print("    finmag (Netgen, this env): {} nodes, {} tets".format(
        fnodes.shape[0], fcells.shape[0]))

    mnodes, mflat = magpar_io.get_field(
        os.path.join(ANIS_DIR, "magpar_result", "test_anis"), "anis")
    mcells = magpar_io.read_femsh(
        os.path.join(ANIS_DIR, "magpar_result", "test_anis.0001.femsh"))[1]
    if mcells.min() == 1:                # Magpar .femsh is 1-based
        mcells = mcells - 1
    N = mnodes.shape[0]
    mH = np.column_stack((mflat[:N], mflat[N:2 * N], mflat[2 * N:3 * N]))
    minp = magpar_io.read_inp(
        os.path.join(ANIS_DIR, "magpar_result", "test_anis.0001"))
    mM = np.column_stack((minp["M_x"], minp["M_y"], minp["M_z"]))
    mm = mM / np.linalg.norm(mM, axis=1)[:, None]
    print("    magpar (saved 2017 run):   {} nodes, {} tets".format(
        N, mcells.shape[0]))
    print("    -> node counts {}".format(
        "MATCH" if N == fnodes.shape[0] else "DIFFER (mesh regenerated)"))

    # nodal m must be the analytic pattern (CG1 interpolation is nodal)
    an_fm = m_gen(fnodes.T).T
    print("    finmag nodal |m - m_gen(x)| max = {:.3e}".format(
        np.abs(fm - an_fm).max()))
    print("    magpar nodal |m - m_gen(x)| max = {:.3e}".format(
        np.abs(mm - m_gen(mnodes.T).T).max()))

    # ------------------------------------------- A: our field == box method?
    print("\n[A] finmag H  vs  INDEPENDENT box projection on finmag's mesh")
    fbox, fvol = box_average(fnodes, fcells, fm[:, 0])
    pred = PREF * fbox
    scale = np.abs(PREF)
    rel = np.abs(fH[:, 0] - pred) / scale
    stats("|Hx - PREF*boxavg(m_x)| / PREF", rel)
    stats("|Hy|,|Hz| / PREF", np.abs(fH[:, 1:]).max(axis=1) / scale)

    # ------------------------------------- B: our field vs POINTWISE analytic
    print("\n[B] finmag H  vs  POINTWISE analytic H = PREF*(m.a)a")
    an = PREF * an_fm[:, 0]
    relB = np.abs(fH[:, 0] - an) / np.abs(an).max()
    stats("|Hx - PREF*m_x(x_i)| / max|H|", relB)

    # ------------------------------------------ C: Magpar == box method too?
    print("\n[C] Magpar Hani  vs  the SAME box projection on MAGPAR's mesh")
    mbox, mvol = box_average(mnodes, mcells, mm[:, 0])
    mpred = PREF * mbox
    relC = np.abs(mH[:, 0] - mpred) / np.abs(mH).max()
    stats("|Hani_x - PREF*boxavg(m_x)| / max|H|", relC)
    relC_pointwise = np.abs(mH[:, 0] - PREF * mm[:, 0]) / np.abs(mH).max()
    print("\n[D] Magpar Hani  vs  POINTWISE analytic H (on Magpar's nodes)")
    stats("|Hani_x - PREF*m_x(x_i)| / max|H|", relC_pointwise)

    # -------------------------------------- E: coincident-node residual map
    print("\n[E] COINCIDENT NODES (exact nodal lookup, no interpolation)")
    tree = cKDTree(fnodes)
    dist, idx = tree.query(mnodes)
    coinc = dist < COINC_TOL
    ci = np.nonzero(coinc)[0]
    print("    coincident magpar nodes: {} of {}".format(ci.size, N))
    norm = np.abs(mH).max()
    resid = (fH[idx[ci], 0] - mH[ci, 0]) / norm
    stats("|finmag - magpar| / max|H| (coincident)", np.abs(resid))

    # both codes' deviation from the pointwise analytic, same nodes
    devf = (fH[idx[ci], 0] - PREF * mm[ci, 0]) / norm
    devm = (mH[ci, 0] - PREF * mm[ci, 0]) / norm
    stats("finmag deviation from pointwise analytic", np.abs(devf))
    stats("magpar deviation from pointwise analytic", np.abs(devm))

    # patch-centroid-offset prediction of the residual
    fc = patch_centroid(fnodes, fcells)
    mc = patch_centroid(mnodes, mcells)
    g = grad_mx(mnodes[ci])
    pred_resid = PREF * np.einsum(
        "ij,ij->i", g, fc[idx[ci]] - mc[ci]) / norm
    err = np.abs(pred_resid - resid)
    corr = np.corrcoef(pred_resid, resid)[0, 1]
    print("    patch-centroid-offset PREDICTION of the residual:")
    print("        corr(predicted, observed) = {:.4f}".format(corr))
    print("        slope (lstsq)             = {:.4f}".format(
        float(np.linalg.lstsq(pred_resid[:, None], resid, rcond=None)[0][0])))
    stats("|predicted - observed| / max|H|", err)
    print("        max |offset| finmag = {:.3f} nm, magpar = {:.3f} nm".format(
        np.linalg.norm(fc - fnodes, axis=1).max(),
        np.linalg.norm(mc - mnodes, axis=1).max()))

    # why does the MAX land on a coincident node? coincident nodes are the
    # geometric feature points both meshers reproduce -- i.e. corners/edges of
    # the bar, where the one-sided patch has the LARGEST centroid offset.
    on_bnd = ((np.abs(mnodes) < 1e-9) | (np.abs(mnodes - 20.0) < 1e-9))
    n_faces = on_bnd.sum(axis=1)
    off_m = np.linalg.norm(mc - mnodes, axis=1)
    print("    coincident nodes by boundary co-dimension "
          "(0=interior,1=face,2=edge,3=corner):")
    for k in range(4):
        sel = n_faces[ci] == k
        if sel.sum():
            print("        codim {}: {:4d} nodes, mean |centroid offset| = "
                  "{:.3f} nm, mean |residual| = {:.4f}".format(
                      k, int(sel.sum()), off_m[ci][sel].mean(),
                      np.abs(resid)[sel].mean()))
    print("        ALL magpar nodes by codim: mean |centroid offset| = " +
          ", ".join("codim {}: {:.3f} nm".format(
              k, off_m[n_faces == k].mean()) for k in range(4)
              if (n_faces == k).sum()))

    order = np.argsort(-np.abs(resid))[:8]
    print("    worst coincident nodes (x, y, z | observed | predicted):")
    for k in order:
        n = mnodes[ci[k]]
        print("        ({:6.3f},{:6.3f},{:6.3f}) | {:+.4f} | {:+.4f}".format(
            n[0], n[1], n[2], resid[k], pred_resid[k]))

    # ------------------------------------------------- F: m-dependence control
    print("\n[F] CONTROL: uniform m (grad m = 0) -- patch average is exact")
    m_uniform = np.array([0.6, 0.8, 0.0])

    def const_m(r):
        one = np.ones_like(np.asarray(r[0], dtype=float))
        return np.array([m_uniform[0] * one, m_uniform[1] * one,
                         m_uniform[2] * one])

    _, mu_f, Hu = setup_bar_anis(const_m)
    _, uH = Hu.coords_and_values()
    an_u = PREF * m_uniform[0]
    print("    finmag Hx (uniform m) vs pointwise analytic: "
          "max rel = {:.3e}".format(np.abs(uH[:, 0] - an_u).max() / abs(an_u)))
    # cross-mesh: the box average of a CONSTANT field is mesh-independent
    ub_f = box_average(fnodes, fcells, np.full(fnodes.shape[0],
                                               m_uniform[0]))[0]
    ub_m = box_average(mnodes, mcells, np.full(N, m_uniform[0]))[0]
    print("    cross-mesh box-average spread (uniform m): finmag {:.3e}, "
          "magpar {:.3e}".format(np.abs(ub_f - m_uniform[0]).max(),
                                 np.abs(ub_m - m_uniform[0]).max()))
    # cross-mesh with the REAL m: box averages differ although the analytic
    # pointwise values are identical at coincident coordinates
    print("    cross-mesh box-average spread (real m, coincident nodes): "
          "{:.3e} (relative to max|m_x| = {:.3f})".format(
              np.abs(fbox[idx[ci]] - mbox[ci]).max(), np.abs(fm[:, 0]).max()))

    # ------------------------------------------- G: why the exchange sibling
    # holds 9e-8 -- it has NO mesh drift at all (structured BoxMesh
    # reproducing Magpar's node set AND tessellation), so the patch geometry
    # is identical and the O(h |grad m|) term above cancels exactly.
    print("\n[G] SIBLING exchange comparison: is its mesh really identical?")
    import dolfinx.mesh as dm
    from mpi4py import MPI
    exch_dir = os.path.join(REPO, "src", "finmag", "tests", "comparison",
                            "exchange", "magpar_result")
    enodes, ecells = magpar_io.read_femsh(
        os.path.join(exch_dir, "test_exch.0001.femsh"))
    if ecells.min() == 1:
        ecells = ecells - 1
    bmesh = dm.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0],
                                           [10.0, 1.0, 1.0]], [40, 2, 2])
    bnodes = bmesh.geometry.x[:, :3]
    bcells = np.asarray(bmesh.geometry.dofmap).reshape((-1, 4))
    print("    magpar exch mesh: {} nodes, {} tets;  finmag BoxMesh(40,2,2): "
          "{} nodes, {} tets".format(enodes.shape[0], ecells.shape[0],
                                     bnodes.shape[0], bcells.shape[0]))
    d, j = cKDTree(bnodes).query(enodes)
    print("    max node coordinate mismatch: {:.3e} nm".format(d.max()))

    def cellkey(nodes, cells):
        return set(tuple(sorted(tuple(np.round(nodes[c], 9)) for c in cell))
                   for cell in cells)
    same_cells = cellkey(enodes, ecells) == cellkey(bnodes, bcells)
    print("    tessellation (cells as coordinate sets) identical: {}".format(
        same_cells))
    ec = patch_centroid(enodes, ecells)
    bc = patch_centroid(bnodes, bcells)
    print("    max |patch-centroid difference| between the two meshes: "
          "{:.3e} nm  (anisotropy case: {:.3f} nm)".format(
              np.abs(ec - bc[j]).max(),
              np.linalg.norm(fc[idx[ci]] - mc[ci], axis=1).max()))

    print("\n" + "=" * 78)
    print("VERDICT INPUTS: see [A] (our field is the exact box projection),")
    print("[C] (Magpar's stored field is the same box projection on its own")
    print("mesh), [E] (the residual is predicted by the patch-geometry")
    print("difference) and [F] (it vanishes when grad m = 0).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
