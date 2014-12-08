"""
Derives physical quantities from the primary simulation state.

"""
import logging
import numpy as np
import dolfin as df
import finmag.util.consts as consts
from ..field import Field
from effective_field import EffectiveField
from equation import Equation

logger = logging.getLogger(name="finmag")


class Physics(object):
    def __init__(self, mesh, unit_length=1, periodic_bc=None):
        self.mesh = mesh
        self.unit_length = unit_length
        self.S1 = df.FunctionSpace(mesh, "CG", 1, constrained_domain=periodic_bc)
        self.S3 = df.VectorFunctionSpace(mesh, "CG", 1, dim=3, constrained_domain=periodic_bc)

        self.alpha = Field(self.S1, value=0.5, name="alpha")
        self.dmdt = Field(self.S3, name="dmdt")
        self.H = Field(self.S3, name="H")
        self.m = Field(self.S3, name="m")
        self.Ms = Field(self.S1, value=1, name="Ms")
        self._pins = df.MeshFunctionBool(mesh, 0, False)

        self.effective_field = EffectiveField(self.m, self.Ms, self.unit_length)
        self.update = []

        self.eq = Equation(self.m.as_vector(), self.H.as_vector(), self.dmdt.as_vector())
        self.eq.set_alpha(self.alpha.as_vector())
        self.eq.set_gamma(consts.gamma)
        self.eq.set_saturation_magnetisation(self.Ms.as_vector())

    def hooks_scipy(self):
        """
        Methods that scipy calls during time integration.

        """
        return (self.scipy_rhs,
                self.m.from_array)

    def hooks_sundials(self):
        """
        Methods that sundials calls during time integration.

        """
        return (self.sundials_rhs,
                self.sundials_jtimes,
                self.sundials_psetup,
                self.sundials_psolve,
                self.m.from_array)

    def hooks_sundials_parallel(self):
        """
        Methods that sundials calls during time integration when it
        operates in parallel mode.

        TODO: What does parallel sundials need?

        """
        return ()

    @property
    def pins(self):
        return self._pins

    @pins.setter
    def pins(self, value):
        pass

    def scipy_rhs(self, m, t):
        """
        Computes the dm/dt right hand side ODE term, as used by scipy.

        """
        self.m.from_array(m)
        self.solve(t)
        return self.dmdt.as_array()

    def solve(self, t):
        for update in self.update:
            update(t)
        self.effective_field.update(t)
        self.H.set(self.effective_field.H_eff)  # FIXME: remove double book-keeping
        self.eq.solve()

    def sundials_jtimes(self, mp, J_mp, t, m, fy, tmp):
        """
        Computes the Jacobian-times-vector product, as used by sundials cvode.

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

             \\frac{d LLG(m, H)}{dm} = \\frac{\\partial LLG(m, H)}{\\partial m}\
             + [\\frac{\\partial LLG(m, H)}{\\partial H}]\
             [\\frac{\\partial H(m)}{\\partial m}].


        This is a matrix identity, so to make the derivations easier (and since
        we don't need the full Jacobian matrix) we can write the
        Jacobian-times-vector product as a directional derivative:

        .. math::

             J m' = \\frac{d LLG(m + a m',H(m + a m'))}{d a}|_{a=0}

        The code to compute this derivative is in ``equation.cpp`` but you can
        see that the derivative will depend on m, m', H(m), and dH(m+a m')/da
        [which is labelled H' in the code].

        Most of the components of the effective field are linear in m; if that's the case,
        the directional derivative H' is just H(m')

        .. math::

             H' = \\frac{d H(m+a m')}{da} = H(m')

        """
        assert m.shape == self.m.as_array().shape
        assert mp.shape == m.shape
        assert tmp.shape == m.shape

        # First, compute the derivative H' = dH_eff/dt
        self.m.from_array(mp)
        Hp = tmp.view()
        Hp[:] = self.effective_field.compute_jacobian_only(t)

        if not hasattr(self, 'sundials_reuse_jacobean') or not self.sundials_reuse_jacobean:
            if not np.array_equal(self.m.as_array(), m):
                self.m.from_array(m)
                self.effective_field.update(t)

        self.eq.sundials_jtimes_serial(mp, Hp, J_mp)
        return 0

    def sundials_psetup(self, t, m, fy, jok, gamma, tmp1, tmp2, tmp3):
        # Note that some of the arguments are deliberately ignored, but they
        # need to be present because the function must have the correct signature
        # when it is passed to set_spils_preconditioner() in the cvode class.
        if not jok:
            self.m.from_array(m)
            self.sundials_reuse_jacobean = True
        return 0, not jok

    def sundials_psolve(self, t, y, fy, r, z, gamma, delta, lr, tmp):
        # Note that some of the arguments are deliberately ignored, but they
        # need to be present because the function must have the correct signature
        # when it is passed to set_spils_preconditioner() in the cvode class.
        z[:] = r
        return 0

    def sundials_rhs(self, t, y, ydot):
        """
        Computes the dm/dt right hand side ODE term, as used by sundials cvode.

        """
        self.m.from_array(y)
        self.solve(t)
        ydot[:] = self.dmdt.as_array()
        return 0

    def set_slonczewski(J, P, p, d, with_time_update):
        self.eq.slonczewski(d, P, p, 2, 0)
        if with_time_update is not None:
            self.update.append(with_time_update)
