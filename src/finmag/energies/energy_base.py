import abc
import logging
from finmag.util.timings import timings
import dolfin as df
import numpy as np
from finmag.util.consts import mu0
from finmag.util.meshes import mesh_volume

logger = logging.getLogger('finmag')


class EnergyBase(object):
    """
    Computes a field for a given energy functional.

    Is a class that shoud be derived from for particular energies.

    See Exchange and UniaxialAnisotropy for examples.

    These will pass the ``method`` parameter to the EnergyBase class:

    *Arguments*

        name
            a string that is used in log messages and timings. If we
            derive from this class a class ``Exchange``, then
            ``name="Exchange"`` is a good choice.

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

    - 'box-assemble' is a slower version that assembles the H_ex for a given M in every
      iteration.

    - 'box-matrix-numpy' precomputes a matrix g, so that H_ex = g*M

    - 'box-matrix-petsc' is the same mathematical scheme as 'box-matrix-numpy',
      but uses a PETSc linear algebra backend that supports sparse
      matrices, to exploit the sparsity of g (default choice).

    - 'project': does not use the box method but 'properly projects' the exchange field
      into the function space. Should explore whether this works and/or makes any difference
      (other than being slow.) Untested.


    *Example of Usage*

        .. code-block:: python

            m    = 1e-8
            n    = 5
            mesh = Box(0, m, 0, m, 0, m, n, n, n)

            S3  = VectorFunctionSpace(mesh, "Lagrange", 1)
            C  = 1.3e-11 # J/m exchange constant
            M  = project(Constant((Ms, 0, 0)), V) # Initial magnetisation

            exchange = Exchange(C, Ms)
            exchange.setup(S3, M)

            # Print energy
            print exchange.compute_energy()

            # Exchange field
            H_exch = exchange.compute_field()

            # Using 'box-matrix-numpy' method (fastest for small matrices)
            exchange_np = Exchange(V, M, C, Ms, method='box-matrix-numpy')
            H_exch_np = exchange_np.compute_field()

    """
    _supported_methods = ['box-assemble', 'box-matrix-numpy', 'box-matrix-petsc', 'project']

    def __init__(self, method="box-matrix-petsc", in_jacobian=False):

        if not method in self._supported_methods:
            raise ValueError("'method' argument must be one of: {}, but received: '{}'".format(self._supported_methods, method))

        self.in_jacobian = in_jacobian
        self.method = method
        in_jacobian_msg = "in Jacobian" if in_jacobian else "not in Jacobian"
        logger.debug("Creating {} object with method {}, {}.".format(
            self.__class__.__name__, method, in_jacobian_msg))

    def _timingsname(self, functiondescription):
        """Compose and return a string that is used for timing functions."""
        return 'EnergyBase-' + self.__class__.__name__ + '-' + functiondescription

    def setup(self, E_integrand, nodal_E, S3, m, Ms, unit_length=1):
        """Function to be called after the energy object has been constructed.

        *Arguments*

            E
                Dolfin form that computes the total energy (a scalar) as a function
                of M (and maybe Ms) if assembled.

            nodal_E
                Dolfin form that computes the energy density at each node.

            S3
                Dolfin 3d VectorFunctionSpace on which M is defined

            M
                Magnetisation field (normally normalised)

            Ms
                Saturation magnetitsation (scalar, or scalar Dolfin function)

            unit_length
                unit_length of distances in mesh.
        """

        ###license_placeholder###

        timings.start(self._timingsname('setup'))

        self.S3 = S3

        self.m = m  # keep reference to M
        self.Ms = Ms
        self.unit_length = unit_length

        self.v = df.TestFunction(S3)
        self.E = E_integrand
        self.dE_dM = df.Constant(-1.0 / mu0) \
                    * df.derivative(self.E / self.Ms * df.dx, self.m)
        #self.dE_dM = -1 * df.derivative(self.E, M, self.v)
        self.vol = df.assemble(df.dot(self.v, df.Constant([1, 1, 1]))
                               * df.dx).array()
        self.dim = S3.mesh().topology().dim()

        # Needed for energy density
        S1 = df.FunctionSpace(S3.mesh(), "CG", 1)
        w = df.TestFunction(S1)
        self.nodal_vol = df.assemble(w * df.dx, mesh=S3.mesh()).array() \
                * unit_length ** self.dim
        self.nodal_E = nodal_E

        # This is only needed if we want the energy density
        # as a df.Function, in order to e.g. probe.
        self.ED = df.Function(S1)

        # Total volume of mesh (if we need to compute
        # average magnetisation we can integrate over M and
        # divide by this number)
        self.total_vol = mesh_volume(S3.mesh()) * unit_length**self.dim

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
            print "Desired method was {}.".format(self.method)
            raise NotImplementedError("""Only methods currently implemented are
                                    * 'box-assemble',
                                    * 'box-matrix-numpy',
                                    * 'box-matrix-petsc'
                                    * 'project'""")

        timings.stop(self._timingsname('setup'))

    def compute_field(self):
        """
        Compute the field associated with the energy.

         *Returns*
            numpy.ndarray
                The coefficients of the dolfin-function in a numpy array.

        """

        timings.start(self._timingsname('compute_field'))
        Hex = self.__compute_field()
        timings.stop(self._timingsname('compute_field'))
        return Hex

    def compute_energy(self):
        """
        Return the total energy, i.e. energy density intergrated
        over the whole mesh [in units of Joule].

        *Returns*
            Float
                The energy.

        """
        timings.start(self._timingsname('compute_energy'))
        E = df.assemble(self.E * df.dx) * self.unit_length ** self.dim
        timings.stop(self._timingsname('compute_energy'))
        return E

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
        timings.start(self._timingsname('energy_density'))
        nodal_E = df.assemble(self.nodal_E).array() \
                * self.unit_length ** self.dim
        timings.stop(self._timingsname('energy_density'))
        return nodal_E / self.nodal_vol

    def energy_density_function(self):
        """
        Compute the exchange energy density the same way as the
        energy_density function above, but return a Function to
        allow probing.

        *Returns*
            dolfin.Function
                The energy density function object.

        """
        self.ED.vector()[:] = self.energy_density()
        return self.ED

    def __setup_field_numpy(self):
        """
        Linearise dE_dM with respect to M. As we know this is
        linear [at least for exchange, and uniaxial anisotropy?]
        (should add reference to Werner Scholz paper and
        relevant equation for g), this creates the right matrix
        to compute dE_dM later as dE_dM=g*M.  We essentially
        compute a Taylor series of the energy in M, and know that
        the first two terms (for exchange: dE_dM=Hex, and ddE_dMdM=g)
        are the only finite ones as we know the expression for the
        energy.

        """
        g_form = df.derivative(self.dE_dM, self.m)
        self.g = df.assemble(g_form).array()  # store matrix as numpy array

    def __setup_field_petsc(self):
        """
        Same as __setup_field_numpy but with a petsc backend.

        """
        g_form = df.derivative(self.dE_dM, self.m)
        self.g_petsc = df.PETScMatrix()
        df.assemble(g_form, tensor=self.g_petsc)
        self.H_petsc = df.PETScVector()

    def __setup_field_project(self):
        #Note that we could make this 'project' method faster by computing the matrices
        #that represent a and L, and only to solve the matrix system in 'compute_field'().
        #IF this method is actually useful, we can do that. HF 16 Feb 2012
        H_trial = df.TrialFunction(self.V)
        self.a = df.dot(H_trial, self.v) * df.dx
        self.L = self.dE_dM
        self.H_exch_project = df.Function(self.V)

    def __compute_field_assemble(self):
        return df.assemble(self.dE_dM).array() / self.vol

    def __compute_field_numpy(self):
        Mvec = self.m.vector().array()
        H_ex = np.dot(self.g, Mvec)
        return H_ex / self.vol

    def __compute_field_petsc(self):
        self.g_petsc.mult(self.m.vector(), self.H_petsc)
        return self.H_petsc.array() / self.vol

    def __compute_field_project(self):
        df.solve(self.a == self.L, self.H_project)
        return self.H_project.vector().array()


if __name__ == "__main__":
    pass
