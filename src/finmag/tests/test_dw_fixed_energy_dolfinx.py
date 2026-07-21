"""``FixedEnergyDW`` curated by-name deferral (Task 19).

``FixedEnergyDW`` is untested even on legacy master (no
``dw_fixed_energy_test.py`` exists) and legacy's own todo notes record it as
broken (see the rationale in
``src/finmag/energies/dw_fixed_energy.py``). Rather than porting an
unvalidated, admittedly-broken implementation, Task 19 converts it to a
curated by-name deferral: the raw ``import dolfin`` is gone, but constructing
the class always raises :class:`NotImplementedError` naming the class and the
reason (a Task 29 review item). This suite pins that contract and confirms
the module no longer touches legacy ``dolfin`` at import time. [Claude
Sonnet 5]
"""

import sys

import pytest

from finmag.energies import FixedEnergyDW


def test_fixed_energy_dw_export_does_not_load_legacy_dolfin():
    assert FixedEnergyDW.__module__ == "finmag.energies.dw_fixed_energy"
    assert "dolfin" not in sys.modules


def test_fixed_energy_dw_is_a_curated_by_name_deferral():
    with pytest.raises(NotImplementedError, match="FixedEnergyDW"):
        FixedEnergyDW()


def test_fixed_energy_dw_deferral_names_the_treecode_dependency_and_task_29():
    """The deferral message must explain *why*, not just fail generically:
    it names the deferred Treecode demag-solver dependency and flags this as
    a Task 29 review item, matching the other curated by-name deferrals in
    this port (e.g. ``finmag.util.meshes``)."""
    with pytest.raises(NotImplementedError) as excinfo:
        FixedEnergyDW(left=(1, 0, 0), right=(-1, 0, 0), repeat_time=3)
    message = str(excinfo.value)
    assert "Treecode" in message
    assert "Task 29" in message
    assert "untested" in message
