"""Generate a legacy reference for a tiny ``hysteresis_loop`` run (Task 15).

Runs at the immutable FEniCS-2019 oracle, using the legacy *default*
(native Sundials) integrator backend -- the true legacy ``Simulation``
default. When this oracle was generated the DOLFINx port still carried the
temporary SciPy default (Task 8); SR1 P1.3 restored ``"sundials"`` as the
port's public default too, so the consuming test now compares like with like.
The per-stage adaptive step timeline built by ``sim_relax.relax`` (see that
module) is governed entirely by pure-Python scheduler logic independent of
the ODE backend, so the stage *times* are expected to reproduce exactly;
only the relaxed ``m_average`` values are expected to show small
solver/floating-point-noise-level disagreement against the DOLFINx port on
either backend.

Important finding recorded here (see ``transition-notes.org`` Task 15 and
the class docstrings on ``finmag.sim.hysteresis``): legacy's own
``hysteresis``/``hysteresis_loop`` do *not* actually achieve an independent
re-relaxation at every stage after the first one. Once a `relax()` call
inside a `hysteresis()` loop reuses a `Simulation` whose integrator clock is
already far from zero, the very first scheduled trigger of the *next*
`relax()` call coincides with the integrator's current time (`t ==
integrator.cur_t`), and `Scheduler.run()` explicitly skips integrating in
that case (see the comment in `sim.py`/`scheduler.py`) -- so subsequent
stages "converge" almost immediately without the magnetisation actually
having time to respond to the new field. The legacy test suite itself
documents this exact symptom with a wink (see
``src/finmag/sim/hysteresis_test.py::test_hysteresis_loop_and_plotting``:
"Check that the magnetisation values are as trivial as we expect them to be
;-)", asserting ``m_vals`` stays at ``1.0`` for a whole H-reversing loop).
This fixture pins that same (surprising, but faithfully legacy) behavior for
a case that *does* include uniaxial anisotropy and a field reversal well
past the Stoner-Wohlfarth coercive field -- i.e. even though the field
becomes strongly enough reversed to flip the macrospin in a *fresh*
`relax()` call (see ``sim/hysteresis_test.py``'s standalone
Stoner-Wohlfarth witness), it does *not* flip when reached via
``hysteresis()``'s stage-2-onward relax() calls. This is preserved
verbatim, not "fixed".
"""
import json
import os
import sys

import numpy as np
import dolfin as df

from finmag import Simulation
from finmag.energies import UniaxialAnisotropy

Ms = 8.6e5
UNIT_LENGTH = 1e-9
K1 = 1.0e4
ALPHA = 1.0
STOPPING_DMDT = 1.0
M_INIT = (1.0, 0.05, 0.0)
H_VALS = [(1.0e5, 0.0, 0.0), (33333.0, 0.0, 0.0),
          (-33333.0, 0.0, 0.0), (-1.0e5, 0.0, 0.0)]


def main():
    generator_argv = [
        "dev/bin/run-legacy-oracle",
        "--",
        "pixi",
        "run",
        "--locked",
        "env",
        "PYTHONPATH=src",
        "python",
        os.path.abspath(__file__),
    ]

    mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(5, 5, 5), 1, 1, 1)
    sim = Simulation(mesh, Ms, unit_length=UNIT_LENGTH,
                     name="gen_hysteresis_oracle")
    sim.set_m(M_INIT)
    sim.alpha = ALPHA
    sim.add(UniaxialAnisotropy(K1, (1.0, 0.0, 0.0)))

    stages = []

    def fun(sim):
        stages.append({
            "t": sim.t,
            "m_average": [float(v) for v in sim.m_average],
        })
        return sim.m_average

    sim.hysteresis(H_VALS, fun=fun, stopping_dmdt=STOPPING_DMDT)

    doc = {
        "schema_version": 1,
        "note": (
            "Legacy FEniCS-2019 reference for a tiny 4-stage "
            "Simulation.hysteresis() run (native Sundials backend, the "
            "true legacy default), generated at the frozen oracle commit "
            "via dev/bin/run-legacy-oracle. Not coordinate-ordered nodal "
            "data (there is only one cell/vertex-set on this tiny mesh, "
            "the quantity of interest is the scalar-reduced m_average per "
            "stage): 'stages' holds one entry per H_VALS field, each with "
            "the simulation clock and volume-averaged magnetisation "
            "reached when relax() decided to stop. See this file's module "
            "docstring for the important finding that stages after the "
            "first do not actually re-relax (a genuine, self-acknowledged "
            "legacy limitation of the hysteresis()/relax() interaction, "
            "not a DOLFINx-port regression)."
        ),
        "oracle": {
            "commit": "ba9280934e188d7f3800e7b9865e70a9422f7687",
            "command": generator_argv,
            "generator": "src/finmag/tests/fixtures/gen_hysteresis_oracle.py",
            "generator_note": (
                "Runs the frozen legacy Simulation.hysteresis() with the "
                "native Sundials backend (Simulation's true default) on a "
                "single-cell box mesh with uniaxial anisotropy; per-stage "
                "m_average and simulation time are recorded via the `fun` "
                "callback."
            ),
        },
        "physical_parameters": {
            "Ms": {"value": Ms, "unit": "A/m"},
            "unit_length": {"value": UNIT_LENGTH, "unit": "m"},
            "K1": {"value": K1, "unit": "J/m**3"},
            "easy_axis": {"value": [1.0, 0.0, 0.0], "unit": "1"},
            "alpha": {"value": ALPHA, "unit": "1"},
            "m_init": {"value": list(M_INIT), "unit": "1"},
            "stopping_dmdt": {"value": STOPPING_DMDT, "unit": "degree/ns"},
        },
        "mesh": {
            "recipe": "dolfin.BoxMesh(Point(0,0,0),Point(5,5,5),1,1,1)",
            "parameters": {"x0": [0, 0, 0], "x1": [5, 5, 5],
                            "nx": 1, "ny": 1, "nz": 1},
        },
        "H_vals": [list(h) for h in H_VALS],
        "stages": [
            {
                "t": stage["t"],
                "m_average": stage["m_average"],
                "tolerances": {"absolute": 2e-3, "relative": 2e-3},
            }
            for stage in stages
        ],
    }

    json.dump(doc, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
