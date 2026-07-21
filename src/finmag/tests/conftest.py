"""Shared test hermeticity fixtures for the ``src/finmag/tests`` suite.

``finmag.util.configuration.get_configuration`` reads a hardcoded homedir
path list (``CONFIGURATION_FILES``: ``~/.finmagrc``, ``~/.finmag/finmagrc``)
with no environment-variable override. ``FKDemag.__init__`` consults this
configuration for a ``[demag] solver_type`` fallback whenever the kwarg is
not given, so *every* unpatched ``FKDemag()``/``Demag()``/
``sim_with(demag_solver="FK")`` call in the suite would otherwise consult the
developer's real ``~/.finmagrc``. A stray ``[demag] solver_type`` entry there
(e.g. ``LU``, which the port deliberately rejects by name) would make
unrelated tests fail confusingly on a machine-specific file the suite never
intended to touch.

This autouse fixture monkeypatches ``CONFIGURATION_FILES`` to an empty list
for every test in this directory by default, so the suite never reads a
developer's real config unless a test opts back in explicitly (as
``test_fk_demag_dolfinx.py::test_finmagrc_solver_type_option_raises_by_name``
does, by monkeypatching its own temp-file value after this fixture runs --
the later, per-test ``monkeypatch.setattr`` simply wins for that test, and
``monkeypatch``'s LIFO teardown unwinds both patches cleanly). [Claude Sonnet 5]
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_finmagrc(monkeypatch):
    from finmag.util import configuration

    monkeypatch.setattr(configuration, "CONFIGURATION_FILES", [])
