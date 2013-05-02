import numpy as np
import pytest
import os
from glob import glob
from fileio import *


def test_Table_writer_and_reader(tmpdir):
    os.chdir(str(tmpdir))
    import finmag
    import dolfin as df

    xmin, ymin, zmin = 0, 0, 0    # one corner of cuboid
    xmax, ymax, zmax = 6, 6, 11   # other corner of cuboid
    nx, ny, nz = 3, 3, 6         # number of subdivisions (use ~2nm edgelength)
    mesh = df.BoxMesh(xmin, ymin, zmin, xmax, ymax, zmax, nx, ny, nz)
    # standard Py parameters
    sim = finmag.sim_with(mesh, Ms=0.86e6, alpha=0.5, unit_length=1e-9,
                          A=13e-12, m_init=(1, 0, 1))

    filename = 'test-save_averages-data.ndt'
    ndt = Tablewriter(filename, sim)
    times = np.linspace(0, 3.0e-11, 6 + 1)
    for i, time in enumerate(times):
        print("In iteration {}, computing up to time {}".format(i, time))
        sim.run_until(time)
        ndt.save()

    # now open file for reading
    data = Tablereader(filename)
    print data.timesteps() - times
    print("III")
    assert np.all(np.abs(data.timesteps() - times)) < 1e-25
    mx, my, mz = sim.m_average
    assert abs(data['m_x'][-1] - mx) < 1e-11
    assert abs(data['m_y'][-1] - my) < 1e-11
    assert abs(data['m_z'][-1] - mz) < 1e-11

    # Try reading multiple columns at once by indexing 'data'
    # with multiple indices (in the assert statement).
    dmx = data['m_x']
    dmy = data['m_y']
    dmz = data['m_z']
    dm = np.vstack([dmx, dmy, dmz]).T  # stack the arrays together
    assert np.allclose(dm, np.array(data['m_x', 'm_y', 'm_z']).T)


def test_field_saver(tmpdir):
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


if __name__ == "__main__":
    test_Table_writer_and_reader()
    test_field_saver()
