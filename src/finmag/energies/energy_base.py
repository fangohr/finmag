import logging
import dolfin as df
import numpy as np
from aeon import mtimed
from finmag.util.consts import mu0
from finmag.util.meshes import nodal_volume
from finmag.util import helpers

logger = logging.getLogger('finmag')


class EnergyBase(object):
    """
    Computes a field for a given energy functional.

    Is a class that particular energies should derive from. The derived classes
    should fill in the necessary details and call methods on the parent.
    See Exchange and UniaxialAnisotropy for examples.

    *Arguments*

        method
            possible methods are
                * 'box-assemble'
                * 'box-matrix-numpy'
                * 'box-matrix-petsc' [Default]
                * 'project'

        in_jacobian
            True or False -- decides whether the interaction is included in
            the Jacobian.

    At the moment, we think (all) 'box' methods work
    (and the method is used in Magpar and Nmag).

    - 'box-assemble' is a slower version that assembles the field H for a
      given m in every iteration.

    - 'box-matrix-numpy' precomputes a matrix g, so that H = g * m.

    - 'box-matrix-petsc' is the same mathematical scheme as 'box-matrix-numpy',
      but uses a PETSc linear algebra backend that supports sparse matrices
      to exploit the sparsity of g (default choice).

    - 'project': does not use the box method but 'properly projects' the
      field into the function space. This is provided for reasons of
      completeness but is potentially untested.

    """
    _supported_methods = ['box-assemble', 'box-matrix-numpy', 'box-matrix-petsc', 'project']

    def __init__(self, method="box-matrix-petsc", in_jacobian=False):

        if not method in self._supported_methods:
            logger.error("Can't create '{}' object with method '{}'. Possible choices are {}.".format(
                self.__class__.__name__, method, self._supported_methods))
            raise ValueError("Unsupported method '{}' should be one of {}.".format(
                method, self._supported_methods))
        else:
            logger.debug("Creating {} object with method {}, {}in Jacobian.".format(
                self.__class__.__name__, method, "not " if not in_jacobian else ""))

        self.in_jacobian = in_jacobian
        self.method = method

    def setup(self, E_integrand, S3, m, Ms, unit_length=1):
        """
        Function to be called after the energy object has been constructed.

        *Arguments*

            E_integrand
                dolfin form that represents the term inside the energy
                integral, as a function of m (and maybe Ms) if assembled

            S3
                Dolfin 3d VectorFunctionSpace on which m is defined

            m
                magnetisation field (usually normalised)

            Ms
                Saturation magnetisation (scalar, or scalar dolfin function)

            unit_length
                real length of 1 unit in the mesh

        """
        ###license_placeholder###

        self.E_integrand = E_integrand
        dofmap = S3.dofmap()
        self.S1 = df.FunctionSpace(S3.mesh(), "Lagrange", 1, constrained_domain=dofmap.constrained_domain)
        self.S3 = S3
        self.m = m # keep reference to m
        self.Ms = Ms
        self.unit_length = unit_length

        self.E = E_integrand * df.dx
        self.nodal_E = df.dot(E_integrand, df.TestFunction(self.S1)) * df.dx
        self.dE_dm = df.Constant(-1.0 / mu0) \
                * df.derivative(E_integrand / self.Ms * df.dx, self.m)


        self.dim = S3.mesh().topology().dim()
        self.nodal_volume_S1 = nodal_volume(self.S1, self.unit_length)
        # same as nodal_volume_S1, just three times in an array to have the same
        # number of elements in the array as the field to be able to divide it.
        self.nodal_volume_S3 = nodal_volume(self.S3)


        if self.method == 'box-assemble':
            self.__compute_field = self.__compute_field_assemble
        elif self.method == 'box-matrix-numpy':
            self.__setup_field_numpy()
            self.__compute_field = self.__compute_field_numpy
        elif self.method == 'box-matrix-petsc':
            self.__setup_field_petsc()
            self.__compute_field = self.__compute_field_petsc
        elif self.method == 'project':
            self.__setup_field_project()
            self.__compute_field = self.__compute_field_project
        else:
            logger.error("Can't create '{}' object with method '{}'. Possible choices are {}.".format(
                self.__class__.__name__, self.method, self._supported_methods))
            raise ValueError("Unsupported method '{}' should be one of {}.".format(
                self.method, self._supported_methods))

    @mtimed
    def compute_energy(self):
        """
        Return the total energy, i.e. energy density intergrated
        over the whole mesh [in units of Joule].

        *Returns*
            Float
                The energy.

        """
        E = df.assemble(self.E) * self.unit_length ** self.dim
        return E

    @mtimed
    def energy_density(self):
        """
        Compute the energy density,

        .. math::

            \\frac{E}{V},

        where V is the volume of each node.

        *Returns*
            numpy.ndarray
                Coefficients of dolfin vector of energy density.

        """
        nodal_E = df.assemble(self.nodal_E).array() * self.unit_length ** self.dim
        return nodal_E / self.nodal_volume_S1

    def energy_density_function(self):
        """
        Compute the exchange energy density the same way as the
        energy_density function above, but return a dolfin function to
        allow probing.

        *Returns*
            dolfin.Function
                The energy density function object.

        """
        if not hasattr(self, "E_density_function"):
            self.E_density_function = df.Function(self.S1)
        self.E_density_function.vector()[:] = self.energy_density()
        return self.E_density_function

    @mtimed
    def compute_field(self):
        """
        Compute the field associated with the energy.

         *Returns*
            numpy.ndarray
                The coefficients of the dolfin-function in a numpy array.

        """

        Hex = self.__compute_field()

        return Hex

    def average_field(self):
        """
        Compute the average field.
        """
        return helpers.average_field(self.compute_field())

    def __compute_field_assemble(self):
        return df.assemble(self.dE_dm).array() / self.nodal_volume_S3

    def __setup_field_petsc(self):
        """
        Same as __setup_field_numpy but with a petsc backend.

        """
        g_form = df.derivative(self.dE_dm, self.m)
        self.g_petsc = df.PETScMatrix()
        df.assemble(g_form, tensor=self.g_petsc)
        self.H_petsc = df.PETScVector()

    def __compute_field_petsc(self):
        self.g_petsc.mult(self.m.vector(), self.H_petsc)
        return self.H_petsc.array() / self.nodal_volume_S3

    def __setup_field_numpy(self):
        """
        Linearise dE_dm with respect to m. As we know this is linear (at least
        for exchange, and uniaxial anisotropy? Should add reference to Werner
        Scholz paper and relevant equation for g), this creates the right matrix
        to compute dE_dm later as dE_dm = g * m. We essentially compute a Taylor
        series of the energy in m, and know that the first two terms (for
        exchange: dE_dm = Hex, and ddE_dmdm = g) are the only finite ones as we
        know the expression for the energy.

        """
        g_form = df.derivative(self.dE_dm, self.m)
        self.g = df.assemble(g_form).array()

    def __compute_field_numpy(self):
        Mvec = self.m.vector().array()
        H_ex = np.dot(self.g, Mvec)
        return H_ex / self.nodal_volume_S3

    def __setup_field_project(self):
        #Note that we could make this 'project' method faster by computing the matrices
        #that represent a and L, and only to solve the matrix system in 'compute_field'().
        #IF this method is actually useful, we can do that. HF 16 Feb 2012
        self.a = df.dot(df.TrialFunction(self.S3), df.TestFunction(self.S3)) * df.dx
        self.L = self.dE_dm
        self.H_project = df.Function(self.S3)

    def __compute_field_project(self):
        df.solve(self.a == self.L, self.H_project)
        return self.H_project.vector().array()
