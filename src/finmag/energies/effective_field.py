import logging
import numpy as np
from finmag.energies import Exchange, UniaxialAnisotropy

logger = logging.getLogger(name="finmag")

class EffectiveField(object):
    def __init__(self, mesh):
        self._output_shape = 3 * mesh.num_vertices()
        self.interactions = []
        self._callables = []

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

    def compute(self, t):
        for func in self._callables:
            func(t)

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
