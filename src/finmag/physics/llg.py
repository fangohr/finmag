import logging
import numpy as np
import dolfin as df
import finmag.util.consts as consts
from aeon import default_timer, mtimed
from finmag.physics.effective_field import EffectiveField
from finmag.native import llg as native_llg
from finmag.util import helpers
from finmag.util.meshes import nodal_volume

#default settings for logger 'finmag' set in __init__.py
#getting access to logger here
logger = logging.getLogger(name='finmag')


class LLG(object):
    """
    Solves the Landau-Lifshitz-Gilbert equation.

    The equation reads

    .. math::

        \\frac{d\\vec{M}}{dt} = -\\gamma_{LL} (\\vec{M} \\times \\vec{H}) - \\alpha \\gamma_{LL} (\\vec{M} \\times [ \\vec{M} \\times \\vec{H}])

    where :math:`\\gamma_{LL} = \\frac{\\gamma}{1+\\alpha^2}`. In our code
    :math:`-\\gamma_{LL}` is referred to as *precession coefficient* and
    :math:`-\\alpha\\gamma_{LL}` as *damping coefficient*.

    """
    @mtimed
    def __init__(self, S1, S3, do_precession=True, average=False, unit_length=1):
        """
        S1 and S3 are df.FunctionSpace and df.VectorFunctionSpace objects,
        and the boolean do_precession controls whether the precession of the
        magnetisation around the effective field is computed or not.

        """
        logger.debug("Creating LLG object.")
        self.S1 = S1
        self.S3 = S3
        self.mesh = S1.mesh()
        self.DG = df.FunctionSpace(self.mesh, "DG", 0)

        self.set_default_values()
        self.do_precession = do_precession
        self.unit_length = unit_length
        self.do_slonczewski = False
        self.do_zhangli = False
        self.effective_field = EffectiveField(S3, self._m, self.Ms, self.unit_length)
        self.Volume = None  # will be computed on demand, and carries volume of the mesh

    def set_default_values(self):
        self.alpha = df.Function(self.S1)
        self.alpha.assign(df.Constant(0.5))
        self.alpha.rename('alpha', 'Gilbert damping constant')

        self.gamma = consts.gamma
        self.c = 1e11  # 1/s numerical scaling correction \
        #               0.1e12 1/s is the value used by default in nmag 0.2
        self._Ms_dg = df.Function(self.DG)
        self.Ms = 8.6e5  # A/m saturation magnetisation
        self._m = df.Function(self.S3)
        # Arguments to _m.rename() below: (new_short_name, new_long_name).
        # These get displayed e.g. in Paraview when loading an
        # exported VTK file.
        self._m.rename("m", "magnetisation")
        self.pins = []  # nodes where the magnetisation gets pinned

        self._dmdt = df.Function(self.S3)

    def set_pins(self, nodes):
        """
        Hold the magnetisation constant for certain nodes in the mesh.

        Pass the indices of the pinned sites as *nodes*. Any type of sequence
        is fine, as long as the indices are between 0 (inclusive) and the highest index.
        This means you CANNOT use python style indexing with negative offsets counting
        backwards.

        """
        if len(nodes) > 0:
            nb_nodes_mesh = len(self._m.vector().array()) / 3
            if min(nodes) >= 0 and max(nodes) < nb_nodes_mesh:
                self._pins = np.array(nodes, dtype="int")
            else:
                logger.error("Indices of pinned nodes should be in [0, {}), were [{}, {}].".format(nb_nodes_mesh, min(nodes), max(nodes)))
        else:
            self._pins = np.array([], dtype="int")

    def pins(self):
        return self._pins
    pins = property(pins, set_pins)

    def set_alpha(self, value):
        """
        Set the damping constant :math:`\\alpha`.

        The parameter `value` can have any of the types accepted by the
        function :py:func:`finmag.util.helpers.scalar_valued_function` (see its
        docstring for details).

        """
        self.alpha = helpers.scalar_valued_function(value, self.S1)
        self.alpha.rename('alpha', 'Gilbert damping constant')

    @property
    def Ms(self):
        return self._Ms_dg

    @Ms.setter
    def Ms(self, value):
        # XXX TODO: Rename _Ms_dg to _Ms because it is not a DG0 function!!!
        # We need a DG function here, so we should use scalar_valued_dg_function
        dg_fun = helpers.scalar_valued_dg_function(value, self.DG)
        self._Ms_dg.vector().set_local(dg_fun.vector().get_local())
        #FIXME: change back to DG space.
        #self._Ms_dg=helpers.scalar_valued_function(value, self.S1)
        self._Ms_dg.rename('Ms', 'Saturation magnetisation')
        self.volumes = df.assemble(df.TestFunction(self.S1) * df.dx)
        print self.volumes
        Ms = df.assemble(self._Ms_dg*df.TestFunction(self.S1)* df.dx).array()/self.volumes.array()
        self._Ms = Ms.copy()
        self.Ms_av = np.average(self._Ms_dg.vector().array())

    @property
    def M(self):
        """The magnetisation, with length Ms."""
        #FIXME:error here
        m = self.m.view().reshape((3, -1))
        Ms = self.Ms.vector().array() if isinstance(self.Ms, df.Function) else self.Ms
        M = Ms * m
        return M.ravel()

    @property
    def M_average(self):
        """The average magnetisation, computed with m_average()."""
        volume_Ms = df.assemble(self._Ms_dg * df.dx, mesh=self.mesh)
        volume = df.assemble(self._Ms_dg * df.dx, mesh=self.mesh)
        return self.m_average * volume_Ms / volume

    @property
    def m(self):
        """The unit magnetisation."""
        return self._m.vector().array()
    
    @m.setter
    def m(self, value):
        # Not enforcing unit length here, as that is better done
        # once at the initialisation of m.
        self._m.vector().set_local(value)

    @property
    def dmdt(self):
        """ dmdt values for all mesh nodes """
        return self._dmdt.vector().array()
        
    @property
    def sundials_m(self):
        """The unit magnetisation."""
        return self._m.vector().array()
    
    @sundials_m.setter
    def sundials_m(self, value):
        # used to copy back from sundials cvode
        self._m.vector()[:] = value
        
    

    def m_average_fun(self,dx=df.dx):
        """
        Compute and return the average polarisation according to the formula
        :math:`\\langle m \\rangle = \\frac{1}{V} \int m \: \mathrm{d}V`

        """

        mx = df.assemble(self._Ms_dg * df.dot(self._m, df.Constant([1, 0, 0])) * dx)
        my = df.assemble(self._Ms_dg * df.dot(self._m, df.Constant([0, 1, 0])) * dx)
        mz = df.assemble(self._Ms_dg * df.dot(self._m, df.Constant([0, 0, 1])) * dx)
        volume = df.assemble(self._Ms_dg * dx, mesh=self.mesh)

        return np.array([mx, my, mz]) / volume
    m_average=property(m_average_fun)


    def set_m(self, value, normalise=True, **kwargs):
        """
        Set the magnetisation (if `normalise` is True, it is automatically
        normalised to unit length).

        `value` can have any of the forms accepted by the function
        'finmag.util.helpers.vector_valued_function' (see its
        docstring for details).

        You can call this method anytime during the simulation. However, when
        providing a numpy array during time integration, the use of
        the attribute m instead of this method is advised for performance
        reasons and because the attribute m doesn't normalise the vector.

        """
        self.m = helpers.vector_valued_function(value, self.S3, normalise=normalise, **kwargs).vector().array()


    def solve_for(self, m, t):
        self.m = m
        value = self.solve(t)
        return value

    def solve(self, t):
        self.effective_field.update(t)  # we don't use self.effective_field.compute(t) for performance reasons
        H_eff = self.effective_field.H_eff  # alias (for readability)
        H_eff.shape = (3, -1)

        default_timer.start("solve", self.__class__.__name__)
        # Use the same characteristic time as defined by c
        char_time = 0.1 / self.c
        # Prepare the arrays in the correct shape
        m = self.m
        m.shape = (3, -1)

        dmdt = np.zeros(m.shape)
        
        # Calculate dm/dt
        if self.do_slonczewski:
            if self.fun_slonczewski_time_update != None:
                J_new = self.fun_slonczewski_time_update(t)
                self.J[:] = J_new
            native_llg.calc_llg_slonczewski_dmdt(
                m, H_eff, t, dmdt, self.pins,
                self.gamma, self.alpha.vector().array(),
                char_time,
                self.J, self.P, self.d, self._Ms, self.p)
        elif self.do_zhangli:
            H_gradm = self.compute_gradient_field()
            H_gradm.shape=(3,-1)
            native_llg.calc_llg_zhang_li_dmdt(
                m, H_eff, H_gradm, t, dmdt, self.pins,
                self.gamma, self.alpha.vector().array(),
                char_time,
                self.u0, self.beta, self._Ms)
            H_gradm.shape=(-1,)
        else:
            native_llg.calc_llg_dmdt(m, H_eff, t, dmdt, self.pins,
                                 self.gamma, self.alpha.vector().array(),
                                 char_time, self.do_precession)
        dmdt.shape = (-1,)
        H_eff.shape=(-1,)

        default_timer.stop("solve", self.__class__.__name__)

        self._dmdt.vector().set_local(dmdt)

        return dmdt

    # Computes the dm/dt right hand side ODE term, as used by SUNDIALS CVODE
    def sundials_rhs(self, t, y, ydot):
        ydot[:] = self.solve_for(y, t)
        return 0

    def sundials_psetup(self, t, m, fy, jok, gamma, tmp1, tmp2, tmp3):
        # Note that some of the arguments are deliberately ignored, but they
        # need to be present because the function must have the correct signature
        # when it is passed to set_spils_preconditioner() in the cvode class.
        if not jok:
            self.m = m
            self._reuse_jacobean = True

        return 0, not jok

    def sundials_psolve(self, t, y, fy, r, z, gamma, delta, lr, tmp):
        # Note that some of the arguments are deliberately ignored, but they
        # need to be present because the function must have the correct signature
        # when it is passed to set_spils_preconditioner() in the cvode class.
        z[:] = r
        return 0

    # Computes the Jacobian-times-vector product, as used by SUNDIALS CVODE
    @mtimed
    def sundials_jtimes(self, mp, J_mp, t, m, fy, tmp):
        """
        The time integration problem we need to solve is of type

        .. math::

                 \\frac{d y}{d t} = f(y,t)

        where y is the state vector (such as the magnetisation components for
        all sites), t is the time, and f(y,t) is the LLG equation.

        For the implicite integration schemes, sundials' cvode solver
        needs to know the Jacobian J, which is the derivative of the
        (vector-valued) function f(y,t) with respect to the (components
        of the vector) y. The Jacobian is a matrix.

        For a magnetic system N sites, the state vector y has 3N entries
        (because every site has 3 components). The Jacobian matrix J would
        thus have a size of 3N*3N. In general, this is too big to store.

        Fortunately, cvode only needs the result of the multiplication of some
        vector y' (provided by cvode) with the Jacobian. We can thus store
        the Jacobian in our own way (in particular as a sparse matrix
        if we leave out the demag field), and carry out the multiplication of
        J with y' when required, and that is what this function does.

        In more detail: We use the variable name mp to represent m' (i.e. mprime) which
        is just a notation to distinguish m' from m (and not any derivative).

        Our equation is:

        .. math::

             \\frac{dm}{dt} = LLG(m, H)

        And we're interested in computing the Jacobian (J) times vector (m') product

        .. math::

             J m' = [\\frac{dLLG(m, H)}{dm}] m'.

        However, the H field itself depends on m, so the total derivative J m'
        will have two terms

        .. math::

             \\frac{d LLG(m, H)}{dm} = \\frac{\\partial LLG(m, H)}{\\partial m} + [\\frac{\\partial LLG(m, H)}{\\partial H}] [\\frac{\\partial H(m)}{\\partial m}].


        This is a matrix identity, so to make the derivations easier (and since we don't need the full Jacobian matrix) we can write the Jacobian-times-vector product as a directional derivative:

        .. math::

             J m' = \\frac{d LLG(m + a m',H(m + a m'))}{d a}|_{a=0}


        The code to compute this derivative is in ``llg.cc`` but you can see that the derivative will depend
        on m, m', H(m), and dH(m+a m')/da [which is labelled H' in the code].

        Most of the components of the effective field are linear in m; if that's the case,
        the directional derivative H' is just H(m')

        .. math::

             H' = \\frac{d H(m+a m')}{da} = H(m')


        The actual implementation of the jacobian-times-vector product is in src/llg/llg.cc,
        function calc_llg_jtimes(...), which in turn makes use of CVSpilsJacTimesVecFn in CVODE.
        """
        assert m.shape == self.m.shape
        assert mp.shape == m.shape
        assert tmp.shape == m.shape

        # First, compute the derivative H' = dH_eff/dt
        self.m = mp
        Hp = tmp.view()
        Hp[:] = self.effective_field.compute_jacobian_only(t)

        if not hasattr(self, '_reuse_jacobean') or not self._reuse_jacobean:
        # If the field m has changed, recompute H_eff as well
            if not np.array_equal(self.m, m):
                self.m = m
                self.effective_field.update(t)
            else:
                pass
                #print "This actually happened."
                #import sys; sys.exit()

        m.shape = (3, -1)
        mp.shape = (3, -1)
        Hp.shape = (3, -1)
        J_mp.shape = (3, -1)
        # Use the same characteristic time as defined by c
        char_time = 0.1 / self.c
        native_llg.calc_llg_jtimes(m, self.effective_field.H_eff.reshape((3, -1)), mp, Hp, t, J_mp, self.gamma,
                                   self.alpha.vector().array(), char_time, self.do_precession, self.pins)
        J_mp.shape = (-1, )
        m.shape = (-1,)
        mp.shape = (-1,)
        tmp.shape = (-1,)

        # Nonnegative exit code indicates success
        return 0

    def use_slonczewski(self, J, P, d, p, with_time_update=None):
        """
        Activates the computation of the Slonczewski spin-torque term in the LLG.

        *Arguments*

        J is the current density in A/m^2 as a number, dolfin function,
          dolfin expression or Python function. In the last case the
          current density is assumed to be spatially constant but can
          vary with time. Thus J=J(t) should be a function expecting a
          single variable t (the simulation time) and return a number.

          Note that a time-dependent current density can also be given
          as a dolfin Expression, but a python function should be much
          more efficient.

        P is the polarisation (between 0 and 1). It is defined as P = (x-y)/(x+y),
        where x and y are the fractions of spin up/down electrons).

        d is the thickness of the free layer in m.

        p is the direction of the polarisation as a triple (is automatically normalised to unit length).

        - with_time_update:

             A function of the form J(t), which accepts a time step `t`
             as its only argument and returns the new current density.

             N.B.: For efficiency reasons, the return value is currently
                   assumed to be a number, i.e. J is assumed to be spatially
                   constant (and only varying with time).

        """
        self.do_slonczewski = True
        self.fun_slonczewski_time_update = with_time_update

        if isinstance(J, df.Expression):
            J = df.interpolate(J, self.S1)
        if not isinstance(J, df.Function):
            func = df.Function(self.S1)
            func.assign(df.Constant(J))
            J = func
        self.J = J.vector().array()
        assert P >= 0.0 and P <= 1.0
        self.P = P
        self.d = d
        polarisation = df.Function(self.S3)
        polarisation.assign(df.Constant((p)))
        # we use fnormalise to ensure that p has unit length
        self.p = helpers.fnormalise(polarisation.vector().array()).reshape((3, -1))
        
        
    def compute_gradient_matrix(self):
        """
        compute (J nabla) m , we hope we can use a matrix M such that M*m = (J nabla)m.

        """
        tau = df.TrialFunction(self.S3)
        sigma = df.TestFunction(self.S3)
        
        self.nodal_volume_S3 = nodal_volume(self.S3)*self.unit_length
        
        dim = self.S3.mesh().topology().dim()
        
        ty = tz = 0
        
        tx = self._J[0]*df.dot(df.grad(tau)[:,0],sigma)
        
        if dim >= 2:
            ty = self._J[1]*df.dot(df.grad(tau)[:,1],sigma)
        
        if dim >= 3:
            tz = self._J[2]*df.dot(df.grad(tau)[:,2],sigma)
        
        self.gradM = df.assemble((tx+ty+tz)*df.dx)

        #self.gradM = df.assemble(df.dot(df.dot(self._J, df.nabla_grad(tau)),sigma)*df.dx)
        
    
    def compute_gradient_field(self):

        self.gradM.mult(self._m.vector(), self.H_gradm)
        
        return self.H_gradm.array()/self.nodal_volume_S3
    
        
        
    def use_zhangli(self, J_profile=(1e10,0,0), P=0.5, beta=0.01, using_u0=False):
        """
        if using_u0 = True, the factor of 1/(1+beta^2) will be dropped.
        """
        
        self.do_zhangli = True
        self._J = helpers.vector_valued_function(J_profile, self.S3)
        self.J = self._J.vector().array()
        self.compute_gradient_matrix()
        self.H_gradm = df.PETScVector()
        
        const_e = 1.602176565e-19; #elementary charge in As
        mu_B = 9.27400968e-24; #Bohr magneton
        
        self.P = P
        self.beta = beta
        
        u0 = P*mu_B/const_e #P g mu_B/(2 e Ms) and g=2 for electrons
        
        if using_u0:
            self.u0 = u0
        else:
            self.u0 = u0/(1+beta**2)
        
