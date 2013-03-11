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
    s = FieldSaver('data_npy.npy', incremental=True)
    s.save(a1)
    s.save(a1)
    s.save(a2)  # nothing wrong with saving arrays of different shapes
    assert(len(glob('data_npy*.npy')) == 3)
    assert(np.allclose(a1, np.load('data_npy_000000.npy')))
    assert(np.allclose(a1, np.load('data_npy_000001.npy')))
    assert(np.allclose(a2, np.load('data_npy_000002.npy')))

    with pytest.raises(IOError):
        # Existing files should not be overwritten
        FieldSaver('data_npy.npy', incremental=True)

    # Using 'overwrite=True' should remove existing files
    s = FieldSaver('data_npy.npy', overwrite=True, incremental=True)
    assert(len(glob('data_npy*.npy')) == 0)
    s.save(a1)
    s.save(a1)
    assert(len(glob('data_npy*.npy')) == 2)

    # Tidying up: remove files created so far
    for f in glob('data_npy*.npy'):
        os.remove(f)

    # Non-incremental saving
    s = FieldSaver('data_npy.npy', overwrite=True, incremental=False)
    assert(len(glob('data_npy*.npy')) == 0)
    s.save(a1)
    s.save(a1)
    s.save(a1)
    assert(len(glob('data_npy*.npy')) == 1)

    # Extension is added automatically
    s = FieldSaver('data_npy.foo')
    s.save(a1)
    assert(len(glob('data_npy.foo*.npy')) == 1)
