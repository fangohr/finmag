import dolfin as df

class Anisotropy(object):
    """
    Compute the anisotropy field.

    The magnetocrystalline anisotropy energy for uniaxial
    anisotropy is given by

    .. math::

        E_{\\text{ani}} = \\int_\\Omega \\sum_j K(1 - 
        (\\vec a \\cdot \\vec M_j)^2) dx,

    where :math:`K` is the anisotropy constant, 
    :math:`\\vec a` the easy axis and :math:`\\sum_i \\vec M_i` 
    the discrete approximation of the magnetic polarization.

    *Arguments*
        V
            a dolfin VectorFunctionSpace object.
        M
            the dolfin object representing the magnetisation
        K
            the anisotropy constant (int or float)
        a
            the easy axis (use dolfin.Constant)
        
    *Example of Usage*
        .. code-block:: python

            m    = 1e-8
            n    = 5
            mesh = Box(0, m, 0, m, 0, m, n, n, n)

            V = VectorFunctionSpace(mesh, 'Lagrange', 1)
            K = 520e3 # For Co (J/m3)

            a = Constant((0, 0, 1)) # Easy axis in z-direction
            M = project(Constant((1, 0, 0)), V) # Initial magnetisation

            anisotropy = Anisotropy(V, M, K, a)

            # Print energy
            print anisotropy.compute_energy()

            # Anisotropy field
            H_ani = anisotropy.compute_field()
            
        """   
    
    def __init__(self, V, M, K, a):

        # Local testfunction
        w = df.TestFunction(V)

        # Convert K1 to dolfin.Constant
        K = df.Constant((K))
        
        # Anisotropy energy
        self.E_ani = K*(1 - (df.dot(a, M))**2)*df.dx

        # Gradient
        self.dE_dM = df.derivative(self.E_ani, M)

        # Volume
        self.vol = df.assemble(df.dot(w,
                     df.Constant((1,1,1)))*df.dx).array()

        self.V = V

    def compute_field(self):
        """
        Return the gradient divided by volume according to the *box method*.

        *Returns*
            numpy.ndarray
                The effective field.

        """
        return df.assemble(self.dE_dM).array() / self.vol

    def compute_energy(self):
        """
        Return the anisotropy energy.

        *Returns*
            Float
                The anisotropy energy.

        """
        V = self.V
        return df.assemble(self.E_ani, mesh=V.mesh())

