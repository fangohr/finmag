import logging
import dolfin as df
import numpy as np
from finmag.util.helpers import vector_valued_function
from finmag.energies import Exchange, UniaxialAnisotropy

logger = logging.getLogger(name="finmag")


class EffectiveField(object):
    def __init__(self, S3, average=False):
        self.S3 = S3
        fun = df.Function(S3)
        self._output_shape = fun.vector().size()
        self.H_eff = fun.vector().array()

        self.interactions = []
        self._callables = []  # functions for time update of interactions
        
        self.cell_average = average
        
        if self.cell_average:
            self.fun = fun
            self.dg_v = df.VectorFunctionSpace(S3.mesh(), "DG", 0)
            self.v3 = df.TestFunction(S3)
            self.volumes = df.assemble(df.dot(self.v3, df.Constant([1, 1, 1])) * df.dx).array()
            

    def add(self, field, with_time_update=None):
        """
        Add an interaction (such as Exchange, Anisotropy, Demag).

        *Arguments:*

        interaction

             The interaction to be added.

         with_time_update (optional)

             A function of the form f(t), which accepts a time step
             `t` as its only single parameter and updates the internal
             state of the interaction accordingly.
        """
        self.interactions.append(field)

        if isinstance(field, Exchange):
            self.exchange = field
        if isinstance(field, UniaxialAnisotropy):
            if hasattr(self, "anisotropy"):
                logger.warning("Overwriting the effective_field.anisotropy attribute.")
            self.anisotropy = field

        if with_time_update:
            self._callables.append(with_time_update)

    def update(self, t=None):
        """
        Update the effective field internally so that its value
        reflects the value at time `t`.

        The argument `t` can be omitted if no interaction requires a
        time update.

        """
        if t is not None:
            for func in self._callables:
                func(t)
        else:
            if self._callables != []:
                raise ValueError("Some interactions require a time update, but no time step was given.")

        self.H_eff[:] = 0
        for interaction in self.interactions:
            self.H_eff += interaction.compute_field()

        if self.cell_average:
            self.fun.vector().set_local(self.H_eff)
            H_eff_dg = df.interpolate(self.fun, self.dg_v)
            self.H_eff[:] = df.assemble(df.dot(H_eff_dg, self.v3) * df.dx)/self.volumes

    def compute(self, t=None):
        """
        Compute and return the effective field.

        The argument `t` is only required if one or more interactions
        require a time-update.

        """
        self.update(t)
        return self.H_eff.copy()

    def compute_jacobian_only(self, t):
        """
        Compute and return the total contribution of all interactions
        that are included in the Jacobian.

        """
        for func in self._callables:
            func(t)

        H_eff = np.zeros(self._output_shape)
        for interaction in self.interactions:
            if interaction.in_jacobian:
                H_eff += interaction.compute_field()
        return H_eff

    def total_energy(self):
        """
        Compute and return the total energy contribution of all
        interactions present in the simulation.

        """
        energy = 0.
        for interaction in self.interactions:
            energy += interaction.compute_energy()
        return energy

    def get_interaction(self, interaction_name):
        """
        Returns the interaction object with the given name. Raises a
        ValueError if no (or more than one) matching interaction is
        found.

        """
        matching_interactions = filter(lambda x: x.name == interaction_name,
                                      self.interactions)

        if len(matching_interactions) == 0:
            raise ValueError("Couldn't find interaction of type '{}'. "
                             "Did you mean one of {}?".format(
                    interaction_name, [x.name for x in self.interactions]))

        if len(matching_interactions) > 1:
            raise ValueError("Found more than one interaction with name "
                             "'{}'.".format(interaction_name))

        return matching_interactions[0]

    def remove_interaction(self, interaction_type):
        """
        Removes the interaction object of the given type. Raises a
        ValueError if no (or more than one) matching interaction is
        found.

        """
        interaction = self.get_interaction(interaction_type)
        if interaction is not None:
            self.interactions.remove(interaction)

    def get_dolfin_function(self, interaction_type, region=None):
        interaction = self.get_interaction(interaction_type)
        # TODO: Do we keep the field as a dolfin function somewhere?
        f = vector_valued_function(interaction.compute_field(), interaction.S3)
        return f
