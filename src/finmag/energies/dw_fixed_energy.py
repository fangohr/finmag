"""Curated by-name deferral: ``FixedEnergyDW`` (Task 19, Task 29 review item).

``FixedEnergyDW`` approximates a fixed domain-wall boundary condition by
duplicating the mesh some ``repeat_time`` times along ``x`` on either side,
assigning a saturated ``left``/``right`` magnetisation to each duplicate, and
summing the resulting stray (demag) field contributions at the original mesh's
nodes. Unlike :class:`finmag.energies.thin_film_demag.ThinFilmDemag` (Task
19's other, directly ported class), this is NOT a self-contained port
candidate on the ported DOLFINx stack:

1. **Untested even on legacy master.** There is no ``dw_fixed_energy_test.py``
   (or any test) anywhere in the legacy tree; the only usage is an
   interactive notebook
   (``doc/ipython_notebooks_src/ref-domain-wall-energy-class.ipynb``) whose
   own committed output cell is a Python traceback, and legacy's own todo
   list records: *"[2014-01-16 Thu] The computation with the FixedEnergyDW
   class is broken."* (``doc/ipython_notebooks_src/
   todo-notebooks-to-review-fix-write.txt``). There is no verified legacy
   behavior to port faithfully *to*.
2. **Depends on an already-deferred demag solver.** ``__compute_field`` always
   constructs ``Demag(solver='Treecode')``; the DOLFINx port's own
   ``finmag.energies.demag`` package defers the Treecode/GCR solver variants
   by name (only Fredkin-Koehler, ``solver='FK'``, is ported) -- porting
   ``FixedEnergyDW`` would require *also* porting a BEM solver variant that is
   independently out of scope here.
3. **Round-trips through legacy dolfin-XML on disk.** ``bias_mesh``/
   ``write_xml`` serialise a hand-duplicated mesh to a bespoke ``<dolfin>``
   XML file and immediately re-read it with ``df.Mesh(filename)`` -- a format
   DOLFINx cannot read, and legacy's own ``write_xml`` docstring already
   flags the function as broken ("*this function is broken at the moment*",
   attributed to Weiwei/Max in the source). There is no reference behavior to
   preserve here, only an admittedly-broken implementation detail to
   reinvent.
4. **Uses the raw legacy dolfin ``Function``/vector API directly** (``Ms.
   vector().array()``, ``m.vector().set_local()``) with no
   :class:`finmag.field.Field`-mediated equivalent, so a "faithful"
   translation would require inventing untested semantics rather than
   transcribing verified ones.

Per the Task 19 escalation guidance ("when in doubt between a shaky port and
a curated deferral, prefer the deferral"), this module is a curated by-name
deferral instead: the raw ``import dolfin`` is gone (module-scope import list
below is now empty of it), but constructing :class:`FixedEnergyDW` raises
:class:`NotImplementedError` naming the class and this rationale. This is a
Task 29 review item: a full port would require an accompanying Treecode/GCR
demag solver port (Task 23 territory) plus inventing the legacy XML-mesh-
duplication semantics from scratch, with no way to validate the result
against a working legacy reference. [Claude Sonnet 5]
"""

import logging

log = logging.getLogger(name="finmag")

_DEFERRAL_MESSAGE = (
    "FixedEnergyDW is not ported to DOLFINx (Task 19 decision, Task 29 "
    "review item): it is untested even on legacy master (no "
    "dw_fixed_energy_test.py exists, and legacy's own todo notes record "
    "'the computation with the FixedEnergyDW class is broken'), it always "
    "constructs Demag(solver='Treecode') -- itself a deferred demag "
    "solver variant in this port (finmag.energies.demag only ports 'FK') "
    "-- and it round-trips a hand-duplicated mesh through a bespoke, "
    "legacy-documented-as-broken dolfin-XML writer/reader. There is no "
    "verified legacy behavior to port faithfully to; see the module "
    "docstring in finmag/energies/dw_fixed_energy.py for the full "
    "rationale."
)


class FixedEnergyDW:
    """Curated by-name deferral; see the module docstring for the rationale.

    Constructing this class always raises :class:`NotImplementedError`
    (Task 19 decision, Task 29 review item) rather than silently accepting
    arguments it cannot act on.
    """

    def __init__(self, left=(1, 0, 0), right=(-1, 0, 0), repeat_time=5,
                 name="FixedEnergyDW"):
        del left, right, repeat_time, name
        raise NotImplementedError(_DEFERRAL_MESSAGE)
