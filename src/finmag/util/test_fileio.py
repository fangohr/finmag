import os
import numpy
from fileio import Tablewriter, Tablereader


def test_Table_writer_and_reader(tmpdir=os.getcwd()):
    import finmag
    import dolfin as df

    xmin, ymin, zmin = 0, 0, 0    # one corner of cuboid
    xmax, ymax, zmax = 6, 6, 11   # other corner of cuboid
    nx, ny, nz = 3, 3, 6         # number of subdivisions (use ~2nm edgelength)
    mesh = df.BoxMesh(xmin, ymin, zmin, xmax, ymax, zmax, nx, ny, nz)
    # standard Py parameters
    sim = finmag.sim_with(mesh, Ms=0.86e6, alpha=0.5, unit_length=1e-9,
                          A=13e-12, m_init=(1, 0, 1))

    filename = os.path.join(tmpdir, 'test-save_averages-data.ndt')
    ndt = Tablewriter(filename, sim)
    times = numpy.linspace(0, 3.0e-11, 6 + 1)
    for i, time in enumerate(times):
        print("In iteration {}, computing up to time {}".format(i, time))
        sim.run_until(time)
        ndt.save()

    # now open file for reading
    data = Tablereader(filename)
    print data.time() - times
    print("III")
    assert numpy.all(numpy.abs(data.time() - times)) < 1e-25
    mx, my, mz = sim.m_average
    assert abs(data['m_x'][-1] - mx) < 1e-11
    assert abs(data['m_y'][-1] - my) < 1e-11
    assert abs(data['m_z'][-1] - mz) < 1e-11
    
    try:
        os.remove(filename)
    except:
        pass


if __name__ == "__main__":
    test_Table_writer_and_reader()
