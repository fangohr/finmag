import logging
import dolfin as df
import numpy as np
from finmag.util.consts import mu0
from finmag.util import helpers
from math import pi, cos

log = logging.getLogger("finmag")

class Zeeman(object):
    def __init__(self, H, name='Zeeman', **kwargs):
        """
        Specify an external field (in A/m).

        H can have any of the forms accepted by the function
        'finmag.util.helpers.vector_valued_function' (see its docstring for details).

        """
        self.value = H
        self.name = name
        self.kwargs = kwargs
        self.in_jacobian = False

    def setup(self, S3, m, Ms, unit_length=1):
        self.S3 = S3
        self.m = m
        self.Ms = Ms
        self.unit_length = unit_length
        self.set_value(self.value, **self.kwargs)

    def set_value(self, value, **kwargs):
        """
        Set the value of the field (in A/m).

        `value` can have any of the forms accepted by the function
        'finmag.util.helpers.vector_valued_function' (see its
        docstring for details).

        """
        self.value = value
        self.H = helpers.vector_valued_function(value, self.S3, **self.kwargs)
        self.H.rename('H_ext', 'H_ext')
        self.E = - mu0 * self.Ms * df.dot(self.m, self.H)

    def average_field(self):
        """
        Compute the average applied field.
        """
        return helpers.average_field(self.compute_field())

    def compute_field(self):
        return self.H.vector().array()

    def compute_energy(self, dx=df.dx):
        E = df.assemble(self.E * dx) * self.unit_length**3
        return E


class TimeZeeman(Zeeman):
    def __init__(self, field_expression, t_off=None, name='TimeZeeman'):
        """
        Specify a time dependent external field (in A/m), which gets updated as continuously as possible.

        Pass in a dolfin expression that depends on time. Make sure the time
        variable is called t. It will get refreshed by calls to update.
        The argument t_off can specify a time at which the field will
        get switched off.

        Alternatively, `field_expression` can be a 3-array representing a
        constant field. In this case `t_off` must be specified, otherwise
        a ValuError is raised (this is a safety measure because in this
        case there would be no time update at all, so it's likely that the
        user intended to do something else).

        """
        if isinstance(field_expression, (list, tuple, np.ndarray)):
            field_expression = np.asarray(field_expression)
            if not field_expression.shape == (3,):
                raise ValueError(
                    "If field_expression is not a dolfin expression, it must "
                    "be a 3-array (representing a constant external field)")
            if t_off is None:
                raise ValueError(
                    "The argument 'field_expression' is a constant array, but "
                    "t_off was not specified so there will be no time update "
                    "at all. Use the Zeeman class instead of TimeZeeman if "
                    "this is what you really want.")
            # Convert the array to a dolfin constant so that we can proceed as normal
            field_expression = df.Constant(map(str, field_expression))

        assert isinstance(field_expression, (df.Expression, df.Constant))
        super(TimeZeeman, self).__init__(field_expression, name=name)
        # TODO: Maybe set a 'checkpoint' for the time integrator at
        #       time t_off? (See comment in update() below.)
        self.t_off = t_off
        self.switched_off = False

    def update(self, t):
        if not self.switched_off:
            if self.t_off and t >= self.t_off:
                # TODO: It might be cleaner to explicitly set a
                #       'checkpoint' for the time integrator at time
                #       t_off, otherwise there is the possibility of
                #       it slightly "overshooting" and thus missing
                #       the exact time the field is switched off.
                #       (This should probably happen in __init__)
                self.switch_off()
                return
            self.value.t = t
            self.H = df.interpolate(self.value, self.S3)
            self.H.rename('H_ext', 'H_ext')

    def switch_off(self):
        # It might be nice to provide the option to remove the Zeeman
        # interaction from the simulation altogether (or at least
        # provide an option to do so) in order to avoid computing the
        # Zeeman energy at all once the field is switched off.
        log.debug("Switching external field off.")
        self.H.assign(df.Constant((0, 0, 0)))
        self.value = None
        self.switched_off = True


class DiscreteTimeZeeman(TimeZeeman):
    def __init__(self, field_expression, dt_update=None, t_off=None, name='DiscreteTimeZeeman'):
        """
        Specify a time dependent external field which gets updated in
        discrete time intervals.

        Pass in a dolfin expression that depends on time. Make sure
        the time variable is called t. It will get refreshed by calls
        to update, if more than dt_update time has passed since the
        last refresh. The argument t_off can specify a time at which
        the field will get switched off. If t_off is provided,
        dt_update can be `None` so that the field remains constant
        until it is switched off.

        """
        if dt_update is None and t_off is None:
            raise ValueError("At least one of the arguments 'dt_update' and "
                             "'t_off' must be given.")
        super(DiscreteTimeZeeman, self).__init__(field_expression, t_off, name=name)
        self.dt_update = dt_update
        self.t_last_update = 0.0

    def update(self, t):
        if not self.switched_off:
            if self.t_off and t >= self.t_off:
                self.switch_off()
                return

            if self.dt_update is not None:
                dt_since_last_update = t - self.t_last_update
                if dt_since_last_update >= self.dt_update:
                    self.value.t = t
                    self.H = df.interpolate(self.value, self.S3)
                    self.H.rename('H_ext', 'H_ext')
                    log.debug("At t={}, after dt={}, update external field again.".format(
                        t, dt_since_last_update))


class TimeZeemanPython(TimeZeeman):
    def __init__(self, df_expression, time_fun, t_off=None, name='TimeZeemanPython'):
        """
        Faster version of the TimeZeeman class for the special case
        that only the amplitude (or the direction) of the field
        varies over time. That is, if `H_0` denotes the field at time
        t=0 then the field value at some point `x` at time `t` is
        assumed to be of the form:

           H(t, x) = H_0(x) * time_fun(t)

        In this situation, the dolfin.interpolate method only needs to
        be evaluated once at the beginning for the spatial expression,
        which saves a lot of computational effort.
        
        *Arguments*

        df_expression :  dolfin.Expression

            The dolfin Expression representing the inital field value
            (this can vary spatially but must not depend on time).

        time_fun :  callable

            Function representing the scaling factor for the amplitude at time.
            
            Note that if the given dolfin expression is a scalar,
            then the time_fun have to return a 3d vector, for example,
            a spatial rotational field around x-axis could be expressed as,
            
                Hy = h0(x,y,z)*cos(wt)
                Hz = h0(x,y,z)*sin(wt)

        t_off :  float

            Time at which the field is switched off.
        """
        assert isinstance(df_expression, (df.Expression, df.Constant))
        self.df_expression = df_expression
        self.time_fun = time_fun
        self.t_off = t_off
        self.switched_off = False
        self.name = name
        self.in_jacobian = False
        
        self.scalar_df_expression = False
        if df_expression.value_size()==1:
            self.scalar_df_expression = True

    def setup(self, S3, m, Ms, unit_length=1):
        self.S3 = S3
        self.m = m
        self.Ms = Ms
        self.unit_length = unit_length
        if self.scalar_df_expression:
            #FIXME: we need to construct a function space if the pbc is applied
            self.h0 = helpers.scalar_valued_function(self.df_expression,S3.mesh()).vector().array()
            self.H0 = df.Function(self.S3)
        else:
            self.H0 = helpers.vector_valued_function(self.df_expression, self.S3)
        
        self.E = - mu0 * self.Ms * df.dot(self.m, self.H0)

        self.H_init = self.H0.vector().array()
        self.H = self.H_init.copy()

    def update(self, t):
        if not self.switched_off:
            if self.t_off and t >= self.t_off:
                self.switch_off()
                return
            
            if self.scalar_df_expression:
                tx,ty,tz=self.time_fun(t)
                
                self.H.shape=(3,-1)
                self.H[0,:]=self.h0*tx
                self.H[1,:]=self.h0*ty
                self.H[2,:]=self.h0*tz
                self.H.shape=(-1,)
            else:
                self.H[:] = self.H_init[:]*self.time_fun(t)

    def switch_off(self):
        # It might be nice to provide the option to remove the Zeeman
        # interaction from the simulation altogether (or at least
        # provide an option to do so) in order to avoid computing the
        # Zeeman energy at all once the field is switched off.
        log.debug("Switching external field off.")
        self.H = np.zeros_like(self.H)
        self.value = None
        self.switched_off = True

    def average_field(self):
        """
        Compute the average applied field.
        """
        return helpers.average_field(self.compute_field())

    def compute_field(self):
        return self.H

    def compute_energy(self, dx=df.dx):
        self.H0.vector().set_local(self.H)
        E = df.assemble(self.E * dx) * self.unit_length**3
        return E


class OscillatingZeeman(TimeZeemanPython):
    def __init__(self, H0, freq, phase=0, t_off=None, name='OscillatingZeeman'):
        """
        Create a field which is constant in space but whose amplitude
        varies sinusoidally with the given frequency and phase. More
        precisely, the field value at time t is:

            H(t) = H0 * cos(2*pi*freq*t + phase)

        Where H0 is a constant 3-vector representing the 'base field'.


        *Arguments*

        H0 :  3-vector

            The constant 'base field' which is scaled by the oscillating amplitude.

        freq :  float

            The oscillation frequency.

        phase :  float

            The phase of the oscillation.

        t_off :  float

            Time at which the field is switched off.

        """
        H0_expr = df.Constant(map(str, H0))

        def amplitude(t):
            return cos(2 * pi * freq * t + phase)

        super(OscillatingZeeman, self).__init__(H0_expr, amplitude, t_off=t_off, name=name)

