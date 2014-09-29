"""
Gathers all the ways we know about how to set the values of a dolfin function.

"""
import dolfin as df


def from_constant(function, constant):
    """
    Set function values using dolfin constant.
    
    """
    function.assign(constant)


def from_expression(function, expression):
    """
    Set function values using dolfin expression.

    """
    temp_function = df.interpolate(expression, function.function_space())
    function.vector().set_local(temp_function.vector().get_local())


def from_field(function, field):
    """
    Set function values using instance of Field class.

    """
    if function.function_space() != field.function_space:
        raise ValueError("function spaces do not match")
    function.vector().set_local(field.function.get_local())


def from_function(function, other_function):
    """
    Set function values using another dolfin function.

    """
    if function.function_space() != other_function.function_space():
        raise ValueError("function spaces do not match")
    function.vector().set_local(function.vector().get_local())


def from_iterable(function, iterable):
    """
    Set function values using iterable (like list, tuple, numpy array).

    """
    if isinstance(function.function_space(), df.FunctionSpace):
        pass



