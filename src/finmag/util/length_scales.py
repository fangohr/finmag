import dolfin as df
from finmag.field import Field
from finmag.util.consts import mu0
from numpy import pi


def exchange_length(A, Ms):
    """
    Computes the exchange length when the exchange constant A
    and the saturation magnetisation Ms are given. Both Ms and A
    are Field objects.

    """
    dg_functionspace = df.FunctionSpace(A.mesh(), 'DG', 0)
    exchange_length = Field(dg_functionspace)

    function = df.project(df.sqrt(2*A.f/(mu0*Ms.f**2)), dg_functionspace)
    exchange_length.set(function)

    return exchange_length


def bloch_parameter(A, K1):
    """
    Computes the Bloch parameter when the exchange constant A
    and the anisotropy constant K1 are given. Both A and K1
    are Field objects.

    """
    dg_functionspace = df.FunctionSpace(A.mesh(), 'DG', 0)
    bloch_parameter = Field(dg_functionspace)

    function = df.project(df.sqrt(A.f/K1.f), dg_functionspace)
    bloch_parameter.set(function)

    return bloch_parameter


def helical_period(A, D):
    """
    Computes the helical period when exchange constant A and
    the constant D are given. Both A and D are Field objects.
    """
    dg_functionspace = df.FunctionSpace(A.mesh(), 'DG', 0)
    helical_period = Field(dg_functionspace)

    function = df.project(df.sqrt(4*pi*A.f/D.f), dg_functionspace)
    helical_period.set(function)

    return helical_period
