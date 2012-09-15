import numpy as np
from finmag.energies import Exchange, UniaxialAnisotropy

class EffectiveField(object):
    def __init__(self, mesh):
        self._output_shape = 3 * mesh.num_vertices()
        self.interactions = []
        self._callables = []

    def add(self, field, with_time_update=None):
        self.interactions.append(field)

        if isinstance(field, Exchange):
            self.exchange = field
        if isinstance(field, UniaxialAnisotropy):
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
