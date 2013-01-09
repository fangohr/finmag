import logging
import numpy as np
from finmag.util.helpers import vector_valued_function
from finmag.energies import Exchange, UniaxialAnisotropy, Demag, Zeeman

logger = logging.getLogger(name="finmag")


class EffectiveField(object):
    field_classes = {
        "exchange": Exchange,
        "demag": Demag,
        "anisotropy": UniaxialAnisotropy,
        "zeeman": Zeeman
        }

    def __init__(self, mesh):
        self._output_shape = 3 * mesh.num_vertices()
        self.interactions = []
        self._callables = []  # functions for time update of interactions

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

    def compute(self, t=None):
        if t is not None:
            for func in self._callables:
                func(t)
        else:
            if self._callables != []:
                raise ValueError("Some interactions require a time update, but no time step was given.")

        H_eff = np.zeros(self._output_shape)
        for interaction in self.interactions:
            H_eff += interaction.compute_field()
        self.H_eff = H_eff
        return self.H_eff

    def compute_jacobian_only(self, t):
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

    def get_interaction(self, interaction_type):
        """
        Returns the interaction object of the given type, or raises a ValueError
        if no, or more than one matching interaction is found.

        """
        if not interaction_type in self.field_classes:
            raise ValueError(
                "'interaction_type' must be a string representing one of the "
                "known field types: {}".format(self.field_classes.keys()))

        interactions_of_type = [e for e in self.interactions
                     if isinstance(e, self.field_classes[interaction_type])]

        if not len(interactions_of_type) == 1:
            raise ValueError(
                "Expected one interaction of type '{}' in simulation. "
                "Found: {}".format(interaction_type, len(interactions_of_type)))

        return interactions_of_type[0]

    def get_dolfin_function(self, interaction_type):
        interaction = self.get_interaction(interaction_type)
        # TODO: Do we keep the field as a dolfin function somewhere?
        return vector_valued_function(interaction.compute_field(), interaction.S3)
