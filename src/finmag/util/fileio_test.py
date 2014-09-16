import numpy as np
import pytest
import os
from glob import glob
from fileio import *

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


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

    # Reading an incomplete dataset should raise a runtime error
    with pytest.raises(RuntimeError):
        Tablereader(os.path.join(MODULE_DIR, 'test-incomplete-data.ndt'))


def test_Tablewriter_complains_about_changing_entities():
    import finmag
    sim = finmag.example.barmini(name='tmp-test-fileio2')
    # create ndt file
    sim.save_averages()

    import pytest
    with pytest.raises(RuntimeError):
        # add entity (should raise RuntimeError)
        sim.tablewriter.add_entity(
            'test', {'header': 'test', 'unit': '<>', 'get': lambda s: -1})


def test_Tablewriter_complains_about_incomplete_entities():
    import pytest
    import finmag
    sim = finmag.example.barmini(name='tmp-test-fileio2')

    # this should pass
    sim.tablewriter.add_entity(
        'test', {'header': 'test', 'unit': '<>', 'get': lambda s: -1})

    # this should fail because we add 'test' the second time
    with pytest.raises(AssertionError):
        sim.tablewriter.add_entity(
            'test', {'header': 'test', 'unit': '<>', 'get': lambda s: -1})

    with pytest.raises(AssertionError):
        # should fail because 'header' is missing
        sim.tablewriter.add_entity(
            'test2', {'unknown-keyword': 'test', 'unit': '<>', 'get': lambda s: -1})

    with pytest.raises(AssertionError):
        # should fail because 'unit' is missing
        sim.tablewriter.add_entity(
            'test3', {'header': 'test', 'unit-is-missing': '<>', 'get': lambda s: -1})

    with pytest.raises(AssertionError):
        # should fail because 'get' is missing
        sim.tablewriter.add_entity(
            'test4', {'header': 'test', 'unit': '<>', 'get-is-missing': lambda s: -1})

    with pytest.raises(AssertionError):
        # should fail because dictionary is not complete
        sim.tablewriter.add_entity('test4', {})


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
