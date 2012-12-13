import logging
import dolfin as df
from finmag.util.consts import mu0
from finmag.util import helpers

log = logging.getLogger("finmag")

class Zeeman(object):
    def __init__(self, H, **kwargs):
        """
        Specify an external field.

        H can have any of the forms accepted by the function
        'finmag.util.helpers.vector_valued_function' (see its docstring for details).

        """
        self.value = H
        self.kwargs = kwargs
        self.in_jacobian = False

    def setup(self, S3, m, Ms, unit_length=1):
        self.S3 = S3
        self.m = m
        self.Ms = Ms
        self.set_value(self.value, **self.kwargs)

    def set_value(self, value, **kwargs):
        """
        Set the value of the field.

        `value` can have any of the forms accepted by the function
        'finmag.util.helpers.vector_valued_function' (see its
        docstring for details).

        """
        self.H = helpers.vector_valued_function(value, self.S3, **self.kwargs)
        self.E = - mu0 * self.Ms * df.dot(self.m, self.H) * df.dx 

    def compute_field(self):
        return self.H.vector().array()

    def compute_energy(self):
        E = df.assemble(self.E)
        return E


class TimeZeeman(Zeeman):
    def __init__(self, field_expression, t_off=None):
        """
        Specify a time dependent external field, which gets updated as continuously as possible.
        
        Pass in a dolfin expression that depends on time. Make sure the time
        variable is called t. It will get refreshed by calls to update. 
        The argument t_off can specify a time at which the field will
        get switched off.

        """
        assert isinstance(field_expression, df.Expression)
        super(TimeZeeman, self).__init__(field_expression)
        self.t_off = t_off
        self.switched_off = False

    def update(self, t):
        if not self.switched_off:
            if self.t_off and t >= self.t_off:
                self.switch_off()
                return
            self.value.t = t
            self.H = df.interpolate(self.value, self.S3)

    def switch_off(self):
        log.debug("Switching external field off.")
        self.H.assign(df.Constant((0, 0, 0)))
        self.value = None
        self.switched_off = True


class DiscreteTimeZeeman(TimeZeeman):
    def __init__(self, field_expression, dt_update, t_off=None):
        """
        Specify a time dependent external field, which gets updated in discrete time intervals.
        
        Pass in a dolfin expression that depends on time. Make sure the time
        variable is called t. It will get refreshed by calls to update, if
        more than dt_update time has passed since the last refresh.
        The argument t_off can specify a time at which the field will
        get switched off.

        """
        super(DiscreteTimeZeeman, self).__init__(field_expression, t_off)
        self.dt_update = dt_update
        self.t_last_update = 0.0

    def update(self, t):
        if not self.switched_off:
            if self.t_off and t >= self.t_off:
                self.switch_off()
                return

            dt_since_last_update = t - self.t_last_update
            if dt_since_last_update >= self.dt_update:
                self.value.t = t
                self.H = df.interpolate(self.value, self.S3)
                log.debug("At t={}, after dt={}, update external field again.".format(
                    t, dt_since_last_update))
