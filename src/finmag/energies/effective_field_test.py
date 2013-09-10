import numpy as np
import dolfin as df
import pytest
from finmag.example import barmini
from distutils.version import LooseVersion


def test_effective_field_compute_returns_copy():
    """
    Regression test to ensure that the value returned by
    EffectiveField.compute() does not change as a simulation
    progresses. This used to happen since EffectiveField.compute()
    returned a reference to an internal numpy array instead of a copy.
    Here we check that this is fixed.

    """
    sim = barmini()
    h0 = sim.llg.effective_field.compute()
    h0_copy = h0.copy()
    sim.run_until(1e-12)
    h1 = sim.llg.effective_field.compute()

    assert np.allclose(h0, h0_copy, atol=0, rtol=1e-8)
    assert not np.allclose(h0, h1, atol=0, rtol=1e-8)


@pytest.mark.xfail("LooseVersion(df.__version__) <= LooseVersion('1.2.0')")
def test_mark_regions():
    sim = barmini()
    effective_field = sim.llg.effective_field

    def region_id(pt):
        return 'top' if pt[2] >= 5.0 else 'bottom'

    effective_field.mark_regions(region_id)
    markers = effective_field.region_markers

    id_top = effective_field.region_ids['top']
    id_bottom = effective_field.region_ids['bottom']
    submesh_top = df.SubMesh(sim.mesh, markers, id_top)
    submesh_bottom = df.SubMesh(sim.mesh, markers, id_bottom)

    demag_top = effective_field.get_dolfin_function('Demag', region='top')
    demag_bottom = effective_field.get_dolfin_function('Demag', region='bottom')

    # Check that the retrieved restricted demag field vectors have the expected sizes.
    assert len(demag_top.vector().array()) == 3 * submesh_top.num_vertices()
    assert len(demag_bottom.vector().array()) == 3 * submesh_bottom.num_vertices()
