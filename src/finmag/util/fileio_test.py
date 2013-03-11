import numpy as np
import pytest
import os
from finmag.example import barmini
from glob import glob
from fileio import *


def test_incremental_saver(tmpdir):
    os.chdir(str(tmpdir))

    a1 = np.arange(10)
    a2 = np.arange(20)

    # Test saving to .npz
    s = IncrementalSaver('data_npy.npy')
    s.save(a1)
    s.save(a1)
    s.save(a2)  # nothing wrong with saving arrays of different shapes
    assert(len(glob('data_npy*.npy')) == 3)
    assert(np.allclose(a1, np.load('data_npy_000000.npy')))
    assert(np.allclose(a1, np.load('data_npy_000001.npy')))
    assert(np.allclose(a2, np.load('data_npy_000002.npy')))

    with pytest.raises(IOError):
        # Existing files should not be overwritten
        IncrementalSaver('data_npy.npy')

    # Using 'overwrite=True' should remove existing files
    s = IncrementalSaver('data_npy.npy', overwrite=True)
    assert(len(glob('data_npy*.npy')) == 0)
    s.save(a1)
    s.save(a1)
    assert(len(glob('data_npy*.npy')) == 2)

    with pytest.raises(ValueError):
        IncrementalSaver('data_npz.npz')

    with pytest.raises(ValueError):
        IncrementalSaver('data_vtk.pvd')

    with pytest.raises(ValueError):
        IncrementalSaver('data_hdf5.h5')

    # # Test saving to .npz
    # s = IncrementalSaver('data_npz.npz')
    # s.save(a1)
    # s.save(a1)
    # s.save(a2)  # nothing wrong with saving arrays of different shapes
    # s.save(a2)
    # assert(len(glob('data_npz*.npz')) == 1)
    # f = np.load('data_npz.npz')
    # assert(np.allclose(a1, f['000000']))
    # assert(np.allclose(a1, f['000001']))
    # assert(np.allclose(a2, f['000002']))
    # assert(np.allclose(a2, f['000003']))
    #
    # # Test saving to vtk
    # s = IncrementalSaver('data_vtk.pvd')
    # s.save(a1)
    # s.save(a2)  # nothing wrong with saving arrays of different shapes
    # assert(len(glob('data_vtk*.pvd')) == 1)
    # assert(len(glob('data_vtk*.vtu')) == 2)
    #
    # # Test saving to ndt
    # s = IncrementalSaver('data_averages.ndt')
    # sim = barmini()
    # s.save(sim)
    # s.save(sim)
    # s.save(sim)
    # # XXX TODO: Check that saving the averages worked!
