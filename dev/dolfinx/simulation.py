"""Reduced DOLFINx simulation wrapper for the M5 core-path prototype.

This is intentionally a small composition layer over the checked M4 helpers.
It is not a compatibility implementation of ``finmag.Simulation``; its purpose
is to make the emerging DOLFINx core path reviewable before any public API is
promised. [Codex gpt-5.5 high]
"""

from dataclasses import dataclass
import json
from pathlib import Path

import dolfinx
from dolfinx import mesh
from mpi4py import MPI
import numpy as np

from dev.dolfinx.prototype import average_nodal_vector
from dev.dolfinx.prototype import constant_vector_function
from dev.dolfinx.prototype import cubic_anisotropy_energy
from dev.dolfinx.prototype import dmi_energy
from dev.dolfinx.prototype import exchange_energy
from dev.dolfinx.prototype import explicit_llg_step
from dev.dolfinx.prototype import nodal_vector_values
from dev.dolfinx.prototype import uniaxial_anisotropy_energy
from dev.dolfinx.prototype import vector_function_space
from dev.dolfinx.prototype import zeeman_energy
from dev.dolfinx.relaxation_example import RelaxationParameters
from dev.dolfinx.relaxation_example import SUMMARY_SCHEMA_VERSION
from dev.dolfinx.relaxation_example import parameters_as_summary
from dev.dolfinx.relaxation_example import validate_summary


RESTART_SCHEMA_VERSION = 1
TRACE_SCHEMA_VERSION = 1


@dataclass
class PrototypeSimulation:
    """Minimal DOLFINx simulation object for M5 core-path exploration."""

    domain: object
    magnetisation: object
    parameters: RelaxationParameters
    mesh_label: str = "custom"
    time: float = 0.0

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
        """Return the currently supported M5 prototype energy terms.

        ``dmi`` uses ``dmi_constant`` from ``self.parameters``, which defaults
        to ``0.0``. ``dmi_energy`` only supports 3D meshes, so a non-zero
        ``dmi_constant`` on a 2D mesh (such as ``unit_square``) raises
        explicitly rather than silently doing the wrong thing. [GitHub
        Copilot / Claude Sonnet 5]
        """
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
        dmi = dmi_energy(
            self.magnetisation,
            dmi_constant=self.parameters.dmi_constant,
            unit_length=self.parameters.unit_length,
        )
        cubic_anisotropy = cubic_anisotropy_energy(
            self.magnetisation,
            u1=self.parameters.cubic_anisotropy_u1,
            u2=self.parameters.cubic_anisotropy_u2,
            K1=self.parameters.cubic_anisotropy_K1,
            K2=self.parameters.cubic_anisotropy_K2,
            K3=self.parameters.cubic_anisotropy_K3,
            unit_length=self.parameters.unit_length,
        )
        return {
            "anisotropy": float(anisotropy),
            "cubic_anisotropy": float(cubic_anisotropy),
            "dmi": float(dmi),
            "exchange": float(exchange),
            "total": float(
                exchange + zeeman + anisotropy + dmi + cubic_anisotropy
            ),
            "zeeman": float(zeeman),
        }

    def average_m(self):
        """Return the nodal average magnetisation vector."""
        return average_nodal_vector(self.magnetisation)

    def state_summary(self):
        """Return a JSON-compatible snapshot without advancing the simulation."""
        return {
            "average_m": self.average_m().tolist(),
            "energy_terms": self.energy_terms(),
            "mesh": self.mesh_label,
            "parameters": parameters_as_summary(self.parameters),
            "time": float(self.time),
        }

    def restart_state(self):
        """Return a narrow JSON-compatible restart state for this M5 wrapper.

        This deliberately captures only the reduced prototype state: mesh label,
        parameters, and nodal magnetisation values. It is not a legacy Finmag
        restart format. [Codex gpt-5.5 high]
        """
        return {
            "dolfinx_version": dolfinx.__version__,
            "magnetisation_values": nodal_vector_values(self.magnetisation).tolist(),
            "mesh": self.mesh_label,
            "parameters": parameters_as_summary(self.parameters),
            "schema_version": RESTART_SCHEMA_VERSION,
            "time": float(self.time),
        }

    @classmethod
    def from_restart_state(cls, state):
        """Recreate a reduced simulation from ``restart_state`` output."""
        _validate_restart_state_shape(state)
        nx, ny = _unit_square_dimensions_from_label(state["mesh"])
        sim = cls.unit_square(
            nx=nx,
            ny=ny,
            parameters=_parameters_from_summary(state["parameters"]),
        )

        current_values = nodal_vector_values(sim.magnetisation)
        stored_values = np.asarray(state["magnetisation_values"], dtype=np.float64)
        if stored_values.shape != current_values.shape:
            raise ValueError(
                "restart magnetisation shape %s does not match mesh shape %s"
                % (stored_values.shape, current_values.shape)
            )
        current_values[:] = stored_values
        sim.magnetisation.x.scatter_forward()
        sim.time = float(state["time"])
        return sim

    @classmethod
    def read_restart_state(cls, input_path):
        """Read a reduced restart JSON file and recreate the simulation."""
        return cls.from_restart_state(json.loads(Path(input_path).read_text()))

    def write_restart_state(self, output_path):
        """Write the reduced restart state to JSON on rank 0."""
        state = self.restart_state()
        output_path = Path(output_path)
        if self.domain.comm.rank == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        self.domain.comm.Barrier()
        return state

    def trace_record(self, step):
        """Return one JSON-compatible time-series record for the current state."""
        state = self.state_summary()
        return {
            "average_m": state["average_m"],
            "energy_terms": state["energy_terms"],
            "step": int(step),
            "time": state["time"],
        }

    def step(self, dt, gamma=1.0, alpha=1.0):
        """Advance the reduced simulation by one explicit LLG step."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        explicit_llg_step(
            self.magnetisation,
            effective_field=self.parameters.field,
            dt=dt,
            gamma=gamma,
            alpha=alpha,
        )
        self.time += float(dt)
        return self

    def run_until(self, target_time, dt, gamma=1.0, alpha=1.0):
        """Advance to ``target_time`` with bounded explicit prototype steps.

        This is a deliberately narrow compatibility-shaped probe for the legacy
        ``Simulation.run_until`` concept. It uses the checked explicit nodal
        stepper and records prototype time, but it is not a production DOLFINx
        time integrator. [Codex gpt-5.5 high]
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        target_time = float(target_time)
        if target_time < self.time:
            raise ValueError("target_time must not be before the current time")

        while self.time < target_time:
            self.step(
                dt=min(float(dt), target_time - self.time),
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

    def relaxation_trace(self, steps, dt, gamma=1.0, alpha=1.0):
        """Run relaxation and return JSON-compatible per-step state records.

        The trace is a prototype data-I/O contract for M5. It captures enough
        state for downstream checks without introducing legacy NDT, VTK, or
        scheduler compatibility. [Codex gpt-5.5 high]
        """
        if steps < 1:
            raise ValueError("steps must be positive")
        if dt <= 0:
            raise ValueError("dt must be positive")

        start_time = float(self.time)
        records = [self.trace_record(step=0)]
        for step in range(1, steps + 1):
            self.step(dt=dt, gamma=gamma, alpha=alpha)
            records.append(self.trace_record(step=step))

        return {
            "dolfinx_version": dolfinx.__version__,
            "dt": float(dt),
            "mesh": self.mesh_label,
            "parameters": parameters_as_summary(self.parameters),
            "records": records,
            "schema_version": TRACE_SCHEMA_VERSION,
            "start_time": start_time,
            "steps": int(steps),
            "time": float(self.time),
        }

    def write_relaxation_trace(self, output_path, steps, dt, gamma=1.0, alpha=1.0):
        """Run relaxation and write the reduced per-step trace to JSON."""
        trace = self.relaxation_trace(steps=steps, dt=dt, gamma=gamma, alpha=alpha)
        output_path = Path(output_path)
        if self.domain.comm.rank == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
        self.domain.comm.Barrier()
        return trace

    def relaxation_summary(self, steps, dt, gamma=1.0, alpha=1.0):
        """Run relaxation and return a JSON-compatible reduced summary."""
        energy_history = self.relax(steps=steps, dt=dt, gamma=gamma, alpha=alpha)
        return {
            "dolfinx_version": dolfinx.__version__,
            "dt": float(dt),
            "energy_history": energy_history,
            "final_average_m": self.average_m().tolist(),
            "final_energy": energy_history[-1],
            "final_time": float(self.time),
            "initial_energy": energy_history[0],
            "mesh": self.mesh_label,
            "parameters": parameters_as_summary(self.parameters),
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "steps": int(steps),
        }

    def write_relaxation_summary(self, output_path, steps, dt, gamma=1.0, alpha=1.0):
        """Run relaxation, validate the reduced summary, and write JSON output."""
        summary = validate_summary(
            self.relaxation_summary(steps=steps, dt=dt, gamma=gamma, alpha=alpha)
        )
        output_path = Path(output_path)
        if self.domain.comm.rank == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        self.domain.comm.Barrier()
        return summary


def _validate_restart_state_shape(state):
    """Validate the small restart-state contract before rebuilding a mesh."""
    required_keys = {
        "dolfinx_version",
        "magnetisation_values",
        "mesh",
        "parameters",
        "schema_version",
        "time",
    }
    missing = required_keys.difference(state)
    if missing:
        raise ValueError(
            "restart state is missing keys: %s" % ", ".join(sorted(missing))
        )
    if state["schema_version"] != RESTART_SCHEMA_VERSION:
        raise ValueError("unsupported restart schema version")


def _unit_square_dimensions_from_label(mesh_label):
    """Decode the only mesh labels supported by the reduced restart path."""
    prefix = "unit_square_"
    if not mesh_label.startswith(prefix):
        raise ValueError("only unit-square restart states are supported")

    parts = mesh_label[len(prefix) :].split("x")
    if len(parts) != 2:
        raise ValueError("invalid unit-square mesh label")
    try:
        nx, ny = (int(part) for part in parts)
    except ValueError:
        raise ValueError("invalid unit-square mesh label")
    if nx < 1 or ny < 1:
        raise ValueError("unit-square mesh dimensions must be positive")
    return nx, ny


def _parameters_from_summary(parameters):
    """Rebuild reduced relaxation parameters from JSON-compatible values."""
    required_keys = set(parameters_as_summary(RelaxationParameters()).keys())
    missing = required_keys.difference(parameters)
    if missing:
        raise ValueError(
            "restart parameters are missing keys: %s" % ", ".join(sorted(missing))
        )
    return RelaxationParameters(
        anisotropy_axis=tuple(parameters["anisotropy_axis"]),
        anisotropy_constant=parameters["anisotropy_constant"],
        exchange_constant=parameters["exchange_constant"],
        field=tuple(parameters["field"]),
        saturation_magnetisation=parameters["saturation_magnetisation"],
        unit_length=parameters["unit_length"],
    )
