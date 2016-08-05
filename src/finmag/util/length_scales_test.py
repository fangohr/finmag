import dolfin as df
import numpy as np
import length_scales as ls
from finmag.field import Field
from finmag.util.consts import mu0


class TestLengthScales(object):
    def setup(self):
        # Create a 3d mesh.
        self.mesh3d = df.UnitCubeMesh(11, 10, 10)

        # Create a DG scalar function space.
        self.functionspace = df.FunctionSpace(self.mesh3d,
                                              "DG", 1)

    def test_exchange_length_constant(self):
        A = Field(self.functionspace, 2/mu0)
        Ms = Field(self.functionspace, 1/mu0)

        lex = ls.exchange_length(A, Ms)

        assert np.allclose(lex.get_numpy_array_debug(), 2)

    def test_exchange_length_varying(self):
        A_expression = df.Expression('4/mu0*x[0] + 1e-100', mu0=mu0)
        Ms_expression = df.Expression('2/mu0*x[0] + 1e-100', mu0=mu0)

        A = Field(self.functionspace, A_expression)
        Ms = Field(self.functionspace, Ms_expression)

        lex = ls.exchange_length(A, Ms)

        assert abs(lex.probe((0.5, 0.5, 0.5)) - 2) < 0.05

    def test_bloch_parameter_constant(self):
        A = Field(self.functionspace, 2)
        K1 = Field(self.functionspace, 0.5)

        bloch_parameter = ls.bloch_parameter(A, K1)

        assert np.allclose(bloch_parameter.get_numpy_array_debug(), 2)

    def test_bloch_parameter_varying(self):
        A_expression = df.Expression('4*x[0] + 1e-100')
        K1_expression = df.Expression('x[0] + 1e-100')

        A = Field(self.functionspace, A_expression)
        K1 = Field(self.functionspace, K1_expression)

        bloch_parameter = ls.bloch_parameter(A, K1)

        assert abs(bloch_parameter.probe((0.5, 0.5, 0.5)) - 2) < 0.05

    def test_helical_period_constant(self):
        A = Field(self.functionspace, 1/np.pi)
        D = Field(self.functionspace, 4)

        helical_period = ls.helical_period(A, D)

        assert np.allclose(helical_period.get_numpy_array_debug(), 1)

    def test_helical_period_varying(self):
        A_expression = df.Expression('2/pi*x[0] + 1e-100', pi=np.pi)
        D_expression = df.Expression('8*x[0] + 1e-100')

        A = Field(self.functionspace, A_expression)
        D = Field(self.functionspace, D_expression)

        helical_period = ls.helical_period(A, D)

        assert abs(helical_period.probe((0.5, 0.5, 0.5)) - 1) < 0.05
