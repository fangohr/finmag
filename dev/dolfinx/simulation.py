"""Reduced DOLFINx simulation wrapper for the M5 core-path prototype.

This is intentionally a small composition layer over the checked M4 helpers.
It is not a compatibility implementation of ``finmag.Simulation``; its purpose
is to make the emerging DOLFINx core path reviewable before any public API is
promised. [Codex gpt-5.5 high]
"""

from dataclasses import dataclass

from dolfinx import mesh
from mpi4py import MPI

from dev.dolfinx.prototype import average_nodal_vector
from dev.dolfinx.prototype import constant_vector_function
from dev.dolfinx.prototype import exchange_energy
from dev.dolfinx.prototype import explicit_llg_step
from dev.dolfinx.prototype import uniaxial_anisotropy_energy
from dev.dolfinx.prototype import vector_function_space
from dev.dolfinx.prototype import zeeman_energy
from dev.dolfinx.relaxation_example import RelaxationParameters
from dev.dolfinx.relaxation_example import SUMMARY_SCHEMA_VERSION
from dev.dolfinx.relaxation_example import parameters_as_summary


@dataclass
class PrototypeSimulation:
    """Minimal DOLFINx simulation object for M5 core-path exploration."""

    domain: object
    magnetisation: object
    parameters: RelaxationParameters
    mesh_label: str = "custom"

    @classmethod
    def unit_square(cls, nx=2, ny=2, initial_m=(1.0, 0.0, 0.0), parameters=None):
        """Create a tiny unit-square simulation with uniform magnetisation."""
        if nx < 1 or ny < 1:
            raise ValueError("unit-square mesh dimensions must be positive")
        if parameters is None:
            parameters = RelaxationParameters()

        domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
        function_space = vector_function_space(domain)
        magnetisation = constant_vector_function(function_space, initial_m)
        return cls(
            domain=domain,
            magnetisation=magnetisation,
            parameters=parameters,
            mesh_label="unit_square_%dx%d" % (nx, ny),
        )

    def energy_terms(self):
        """Return the currently supported M5 prototype energy terms."""
        exchange = exchange_energy(
            self.magnetisation,
            exchange_constant=self.parameters.exchange_constant,
            unit_length=self.parameters.unit_length,
        )
        zeeman = zeeman_energy(
            self.magnetisation,
            field=self.parameters.field,
            saturation_magnetisation=self.parameters.saturation_magnetisation,
            unit_length=self.parameters.unit_length,
        )
        anisotropy = uniaxial_anisotropy_energy(
            self.magnetisation,
            axis=self.parameters.anisotropy_axis,
            anisotropy_constant=self.parameters.anisotropy_constant,
            unit_length=self.parameters.unit_length,
        )
        return {
            "anisotropy": float(anisotropy),
            "exchange": float(exchange),
            "total": float(exchange + zeeman + anisotropy),
            "zeeman": float(zeeman),
        }

    def average_m(self):
        """Return the nodal average magnetisation vector."""
        return average_nodal_vector(self.magnetisation)

    def step(self, dt, gamma=1.0, alpha=1.0):
        """Advance the reduced simulation by one explicit LLG step."""
        explicit_llg_step(
            self.magnetisation,
            effective_field=self.parameters.field,
            dt=dt,
            gamma=gamma,
            alpha=alpha,
        )
        return self

    def relax(self, steps, dt, gamma=1.0, alpha=1.0):
        """Run a short deterministic relaxation and return total energy history."""
        if steps < 1:
            raise ValueError("steps must be positive")
        energy_history = [self.energy_terms()["total"]]
        for _ in range(steps):
            self.step(dt=dt, gamma=gamma, alpha=alpha)
            energy_history.append(self.energy_terms()["total"])
        return energy_history

    def relaxation_summary(self, steps, dt, gamma=1.0, alpha=1.0):
        """Run relaxation and return a JSON-compatible reduced summary."""
        energy_history = self.relax(steps=steps, dt=dt, gamma=gamma, alpha=alpha)
        return {
            "dt": float(dt),
            "energy_history": energy_history,
            "final_average_m": self.average_m().tolist(),
            "final_energy": energy_history[-1],
            "initial_energy": energy_history[0],
            "mesh": self.mesh_label,
            "parameters": parameters_as_summary(self.parameters),
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "steps": int(steps),
        }
