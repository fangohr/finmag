"""Fredkin-Koehler hybrid FEM/BEM demagnetising field, ported to DOLFINx.

This is the direct DOLFINx port of the legacy ``FKDemag`` two-potential method.
The discrete formulation is preserved verbatim from the FEniCS-2019 module:

1. ``phi_1`` solves the inhomogeneous Neumann Poisson problem
   ``div(grad(phi_1)) = div(M)`` on the whole domain (pure-Neumann, singular;
   the constant nullspace is projected out -- irrelevant to ``H = -grad(phi)``);
2. the boundary values of ``phi_2`` are set from the boundary element matrix
   ``phi_2|_bnd = B @ phi_1|_bnd`` (the compiled, dolfin-free
   ``finmag.native.bem_arrays.compute_bem_fk_from_arrays`` kernel, Task 11a);
3. ``phi_2`` solves the Laplace problem inside the domain with those Dirichlet
   boundary values;
4. ``H_demag = -grad(phi_1 + phi_2)``, recovered by the same lumped
   box (assemble-and-divide-by-nodal-volume) trick as the legacy module.

Only the FEM-API layer changed (DOLFINx function spaces / PETSc KSP solvers /
boundary submesh extraction / dof maps). The boundary-node ordering fed to the
native BEM kernel is built with an explicit, coordinate-driven mapping and the
boundary triangles are consistently outward-oriented; both are pinned against
the Task 11a golden BEM matrix in ``test_fk_demag_dolfinx.py``.

Deliberate deviations from the legacy module (documented in
``transition-notes.org`` and ``dev/dolfinx/porting_map.md``):

- ``solver_type='LU'`` and ``macrogeometry`` (PBC / ``MacroGeometry``) are
  deferred and raise ``NotImplementedError`` by name; only the Krylov path is
  ported. ``Demag2D`` and the ``Treecode``/``GCR`` solvers likewise raise by
  name through the ``Demag`` factory.
- Serial assembly of the BEM is used (the legacy BEM was effectively serial);
  the FEM solves themselves run through PETSc and are collective, but the
  slice is validated serial-only.

[Claude Opus 4.8]
"""

import logging
from math import pi

import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc

from finmag.field import Field, associated_scalar_space
from finmag.util.configuration import get_config_option

logger = logging.getLogger("finmag")

mu0 = 4.0 * pi * 1e-7


def boundary_bem_arrays(domain, S1):
    """Extract the FK BEM inputs from a DOLFINx volume mesh.

    Returns ``(coords, cells, b2g)`` where

    - ``coords`` is ``(n, 3)`` boundary-node coordinates in BEM-local order,
      one row per CG1 boundary degree of freedom of ``S1``;
    - ``cells`` is ``(m, 3)`` boundary triangles as BEM-local node indices,
      oriented so the triangle normal points *out* of the domain (the winding
      the Lindholm double-layer kernel and the legacy ``BoundaryMesh`` assume);
    - ``b2g`` is ``(n,)`` the S1 degree-of-freedom index for each BEM-local
      boundary node, so ``phi.x.array[b2g]`` gathers boundary values in
      BEM-local order and the reverse scatter writes them back.

    The BEM-local index of a boundary node is defined by its *coordinate*, so
    the mapping is robust to DOLFINx submesh/entity reordering. This is the
    exact place a silent permutation would corrupt the demag field, so it is
    pinned bit-for-bit against the Task 11a golden matrix.
    """
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(fdim, 0)
    domain.topology.create_connectivity(tdim, 0)
    ext_facets = dmesh.exterior_facet_indices(domain.topology)

    boundary_dofs = fem.locate_dofs_topological(S1, fdim, ext_facets)
    boundary_dofs = np.asarray(boundary_dofs, dtype=np.int64)
    dof_coords = S1.tabulate_dof_coordinates()
    coords = np.ascontiguousarray(dof_coords[boundary_dofs], dtype=np.float64)

    n = coords.shape[0]
    keyed = {tuple(np.round(coords[i], 9)): i for i in range(n)}
    if len(keyed) != n:
        raise RuntimeError(
            "FK demag boundary extraction found coincident boundary-node "
            "coordinates; the coordinate mapping would be ambiguous"
        )

    # vertex -> coordinate, via the geometry dofmap (P1 simplex geometry).
    vcoords = domain.geometry.x
    geo_dofmap = domain.geometry.dofmap
    c2v = domain.topology.connectivity(tdim, 0)
    imap = domain.topology.index_map(0)
    nverts = imap.size_local + imap.num_ghosts
    vert_coord = np.full((nverts, 3), np.nan)
    for c in range(c2v.num_nodes):
        vs = c2v.links(c)
        gs = geo_dofmap[c]
        for lv, v in enumerate(vs):
            vert_coord[v] = vcoords[gs[lv]]

    f2v = domain.topology.connectivity(fdim, 0)
    f2c = domain.topology.connectivity(fdim, tdim)
    cells = np.empty((len(ext_facets), 3), dtype=np.int64)
    for row, f in enumerate(ext_facets):
        vs = list(f2v.links(f))
        p = [vert_coord[v] for v in vs]
        cell = f2c.links(f)[0]
        opp = [v for v in c2v.links(cell) if v not in vs][0]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        if np.dot(normal, p[0] - vert_coord[opp]) < 0.0:
            vs = [vs[0], vs[2], vs[1]]
        cells[row] = [keyed[tuple(np.round(vert_coord[v], 9))] for v in vs]

    return coords, cells, boundary_dofs


def _ksp(matrix, method, preconditioner, tol_params, nullspace=None):
    """Build a PETSc KSP for ``matrix`` from legacy-style solver parameters."""
    comm = matrix.getComm()
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(matrix)

    method_map = {
        "default": "cg",
        "cg": "cg",
        "gmres": "gmres",
        "bicgstab": "bcgs",
        "minres": "minres",
        "richardson": "richardson",
    }
    ksp.setType(method_map.get(str(method).lower(), "cg"))

    pc = ksp.getPC()
    pc_map = {
        "default": "hypre",
        "amg": "hypre",
        "hypre": "hypre",
        "petsc_amg": "gamg",
        "gamg": "gamg",
        "ilu": "ilu",
        "jacobi": "jacobi",
        "none": "none",
    }
    pc_type = pc_map.get(str(preconditioner).lower(), "hypre")
    try:
        pc.setType(pc_type)
    except PETSc.Error:  # pragma: no cover - environment-dependent
        pc.setType("gamg")

    rtol = float(tol_params.get("relative_tolerance", 1e-6))
    atol = float(tol_params.get("absolute_tolerance", 1e-6))
    max_it = int(tol_params.get("maximum_iterations", int(1e4)))
    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
    if nullspace is not None:
        matrix.setNullSpace(nullspace)
    ksp.setFromOptions()
    return ksp


class FKDemag(object):
    """Fredkin-Koehler hybrid FEM/BEM demagnetising field (DOLFINx port)."""

    def __init__(self, name="Demag", thin_film=False, macrogeometry=None,
                 solver_type=None, parameters=None):
        self.name = name
        self.in_jacobian = False

        default_parameters = {
            "absolute_tolerance": 1e-6,
            "relative_tolerance": 1e-6,
            "maximum_iterations": int(1e4),
        }
        self.parameters = {
            "phi_1_solver": "default",
            "phi_1_preconditioner": "default",
            "phi_1": default_parameters,
            "phi_2_solver": "default",
            "phi_2_preconditioner": "default",
            "phi_2": default_parameters.copy(),
        }
        if parameters is not None:
            for (k, v) in parameters.items():
                if k in ("phi_1", "phi_2"):
                    for (k2, v2) in v.items():
                        self.parameters[k][k2] = v2
                else:
                    self.parameters[k] = v

        # Mirror the legacy behaviour of falling back to the '.finmagrc'
        # 'demag'/'solver_type' option when the kwarg is not given, so a user
        # config requesting a non-default solver is never silently ignored;
        # a config option other than 'Krylov'/'None' fails loudly by name,
        # exactly like the explicit kwarg path below. [Claude Sonnet 5]
        effective_solver_type = solver_type
        if effective_solver_type is None:
            effective_solver_type = get_config_option(
                'demag', 'solver_type', 'Krylov')
            if effective_solver_type == 'None':
                # A literal 'solver_type = None' in .finmagrc is read back as
                # the string 'None' by configparser; the legacy module treats
                # that the same as not setting it at all.
                effective_solver_type = 'Krylov'

        if str(effective_solver_type).lower() == "lu":
            raise NotImplementedError(
                "FKDemag solver_type='LU' is deferred in the DOLFINx port; "
                "use the Krylov solver (solver_type=None or 'Krylov')"
            )
        if str(effective_solver_type).lower() != "krylov":
            raise NotImplementedError(
                "FKDemag solver_type={!r} (from explicit kwarg or the "
                "'demag'/'solver_type' .finmagrc option) is not implemented "
                "in the DOLFINx port; only the Krylov solver is ported "
                "(solver_type=None or 'Krylov')".format(effective_solver_type)
            )
        self.solver_type = solver_type

        if thin_film:
            self.parameters["phi_1_solver"] = "cg"
            self.parameters["phi_1_preconditioner"] = "ilu"
            self.parameters["phi_2_preconditioner"] = "none"

        if macrogeometry is not None:
            raise NotImplementedError(
                "FKDemag macrogeometry (periodic/MacroGeometry) demag is "
                "deferred to the separate PBC/treecode native slice"
            )
        self.macrogeometry = None

    def setup(self, m, Ms, unit_length=1):
        """Bind the demag solver to magnetisation ``m`` and scalar ``Ms``.

        ``m`` is a three-component CG1 :class:`~finmag.field.Field`; ``Ms`` is a
        scalar :class:`~finmag.field.Field`; ``unit_length`` is the physical
        length (m) of one mesh unit.
        """
        assert isinstance(m, Field)
        assert isinstance(Ms, Field)

        self.m = m
        self.Ms = Ms
        self.unit_length = unit_length
        self.S3 = m.functionspace
        self.domain = m.mesh()
        self.dim = m.mesh_dim()
        self.S1 = associated_scalar_space(self.S3)

        v1 = ufl.TestFunction(self.S1)
        u1 = ufl.TrialFunction(self.S1)
        v3 = ufl.TestFunction(self.S3)

        # lumped nodal volumes (mesh-coordinate units, no unit_length factor).
        self._nodal_volumes_S1 = _assemble_owned(v1 * ufl.dx, self.S1)
        self._nodal_volumes_S3 = _assemble_owned(
            ufl.inner(v3, fem.Constant(self.domain, np.ones(3))) * ufl.dx,
            self.S3,
        )

        # Poisson/Laplace stiffness form, reused for both potentials.
        self._poisson_form = fem.form(
            ufl.inner(ufl.grad(u1), ufl.grad(v1)) * ufl.dx)

        # scalar potentials and the boundary-condition source function.
        self._phi_1 = fem.Function(self.S1)
        self._phi_2 = fem.Function(self.S1)
        self._phi = fem.Function(self.S1)
        self._g_bc = fem.Function(self.S1)  # Dirichlet source for phi_2
        self._H_func = fem.Function(self.S3)  # holds H for energy assembly

        # Boundary element matrix (compiled, dolfin-free) + ordering contract.
        if not hasattr(self, "_bem"):
            from finmag.native.bem_arrays import compute_bem_fk_from_arrays

            coords, cells, b2g = boundary_bem_arrays(self.domain, self.S1)
            self._bem, _ = compute_bem_fk_from_arrays(
                coords, cells, np.asarray(b2g, dtype=np.int64))
            self._b2g_map = np.asarray(b2g, dtype=np.int64)
        logger.debug(
            "Boundary element matrix uses {:.2f} MB of memory.".format(
                self._bem.nbytes / 1024.0 ** 2))

        # linear forms re-assembled every solve (m, phi change).
        self._divergence_form = fem.form(
            self.Ms.f * ufl.inner(self.m.f, ufl.grad(v1)) * ufl.dx)
        self._gradient_form = fem.form(
            ufl.inner(v3, -ufl.grad(self._phi)) * ufl.dx)

        # energy integrands.
        self._E_form = fem.form(
            -0.5 * mu0 * ufl.dot(self._H_func, self.m.f * self.Ms.f) * ufl.dx)
        self._nodal_E_form = fem.form(
            -0.5 * mu0
            * ufl.dot(self._H_func, self.m.f * self.Ms.f) * v1 * ufl.dx)
        self._nodal_E_func = fem.Function(self.S1)

        # Dirichlet BC for phi_2 (values live in self._g_bc, updated per solve).
        self._boundary_dofs = np.asarray(
            self._b2g_map, dtype=np.int32)
        self._bc = fem.dirichletbc(self._g_bc, self._boundary_dofs)

        # Assembled operators.  A_neumann: pure Neumann (phi_1, singular);
        # A_dirichlet: same stiffness with phi_2 boundary rows/cols pinned.
        self._A_neumann = fem_petsc.assemble_matrix(self._poisson_form)
        self._A_neumann.assemble()
        self._nullspace = PETSc.NullSpace().create(
            constant=True, comm=self.domain.comm)
        self._A_dirichlet = fem_petsc.assemble_matrix(
            self._poisson_form, bcs=[self._bc])
        self._A_dirichlet.assemble()

        self._poisson_solver = _ksp(
            self._A_neumann, self.parameters["phi_1_solver"],
            self.parameters["phi_1_preconditioner"], self.parameters["phi_1"],
            nullspace=self._nullspace)
        self._laplace_solver = _ksp(
            self._A_dirichlet, self.parameters["phi_2_solver"],
            self.parameters["phi_2_preconditioner"], self.parameters["phi_2"])

    def precomputed_bem(self, bem, b2g_map):
        """Reuse a previously computed BEM matrix and boundary->global map."""
        self._bem, self._b2g_map = bem, np.asarray(b2g_map, dtype=np.int64)

    def _compute_magnetic_potential(self):
        # phi_1: inhomogeneous Neumann Poisson, div(M) source.
        b1 = fem_petsc.assemble_vector(self._divergence_form)
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                       mode=PETSc.ScatterMode.REVERSE)
        # source is orthogonal to constants; project to be safe.
        self._nullspace.remove(b1)
        self._poisson_solver.solve(b1, self._phi_1.x.petsc_vec)
        self._phi_1.x.scatter_forward()
        b1.destroy()

        # phi_2 boundary values from the BEM, then Laplace solve.
        phi_1_boundary = self._phi_1.x.array[self._b2g_map]
        bem_values = np.dot(self._bem, phi_1_boundary)
        self._g_bc.x.array[:] = 0.0
        self._g_bc.x.array[self._b2g_map] = bem_values
        self._g_bc.x.scatter_forward()

        b2 = fem_petsc.assemble_vector(
            fem.form(fem.Constant(self.domain, 0.0)
                     * ufl.TestFunction(self.S1) * ufl.dx))
        fem_petsc.apply_lifting(b2, [self._poisson_form], bcs=[[self._bc]])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                       mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b2, [self._bc])
        self._laplace_solver.solve(b2, self._phi_2.x.petsc_vec)
        self._phi_2.x.scatter_forward()
        b2.destroy()

        # phi = phi_1 + phi_2
        self._phi.x.array[:] = self._phi_1.x.array + self._phi_2.x.array
        self._phi.x.scatter_forward()

    def compute_potential(self):
        """Compute and return the total magnetic scalar potential Function."""
        self._compute_magnetic_potential()
        return self._phi

    def compute_field(self):
        """Compute the demagnetising field as a flat owned ``xyz`` array."""
        self._compute_magnetic_potential()
        return self._compute_gradient()

    def _compute_gradient(self):
        H = _assemble_owned(None, self.S3, form=self._gradient_form)
        return H / self._nodal_volumes_S3

    def average_field(self):
        """Collective arithmetic average of the demag field over owned nodes."""
        values = self.compute_field().reshape((-1, 3))
        local_sum = np.sum(values, axis=0)
        global_sum = np.zeros_like(local_sum)
        self.domain.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
        count = self.domain.comm.allreduce(values.shape[0], op=MPI.SUM)
        return global_sum / count

    def _load_H_func(self):
        """Copy the current demag field (owned) into ``self._H_func``."""
        owned = self.compute_field()
        self._H_func.x.array[: owned.size] = owned
        self._H_func.x.scatter_forward()

    def compute_energy(self):
        """Total demag energy ``-1/2 mu0 int H.M`` in joules (collective)."""
        self._load_H_func()
        local = fem.assemble_scalar(self._E_form)
        return self.domain.comm.allreduce(local, op=MPI.SUM) \
            * self.unit_length ** self.dim

    def energy_density(self):
        """Owned lumped nodal demag energy density (collective assembly).

        Energy density is intensive: the ``unit_length**dim`` factors on the
        physical nodal energy and the physical nodal volume cancel, so this is
        assembled purely in mesh-coordinate units (matching the legacy module,
        which multiplied *and* divided by ``unit_length**dim``).
        """
        self._load_H_func()
        nodal_E = _assemble_owned(None, self.S1, form=self._nodal_E_form)
        return nodal_E / self._nodal_volumes_S1

    def energy_density_function(self):
        """Return the lumped nodal energy density as a DOLFINx Function."""
        density = self.energy_density()
        self._nodal_E_func.x.array[: density.size] = density
        self._nodal_E_func.x.scatter_forward()
        return self._nodal_E_func


def _owned_scalar_dofs(function_space):
    dofmap = function_space.dofmap
    return dofmap.index_map.size_local * dofmap.index_map_bs


def _assemble_owned(expression, function_space, form=None):
    """Assemble a linear form and return owned entries (ghosts accumulated)."""
    if form is None:
        form = fem.form(expression)
    vec = fem_petsc.assemble_vector(form)
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                    mode=PETSc.ScatterMode.REVERSE)
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,
                    mode=PETSc.ScatterMode.FORWARD)
    owned = np.array(vec.array[: _owned_scalar_dofs(function_space)],
                     dtype=np.float64)
    vec.destroy()
    return owned
