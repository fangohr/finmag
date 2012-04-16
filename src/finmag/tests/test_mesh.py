import numpy as np
import dolfin as df
from finmag.sim.llg import LLG
from finmag.sim.helpers import monkey_patch_mesh

TOL = 1e-15

NANOMETERS = 1e-9
PICOMETERS = 1e-12

x_min = 0; x_max = 1; x_n = 2

def test_mesh_scaled_in_nanometers_by_default():
    mesh = df.Interval(x_n, x_min, x_max)
    llg = LLG(mesh)
    print "Default scaling: {}.".format(llg.mesh.f_scale_factor)
    assert llg.mesh.f_scale_factor == NANOMETERS 

def test_can_overwrite_default_scaling():
    """
    this is achieved by patching the mesh instance before passing
    it to the LLG class.

    """
    mesh = df.Interval(x_n, x_min, x_max)
    monkey_patch_mesh(mesh, PICOMETERS)

    assert mesh.f_scale_factor == PICOMETERS
    llg = LLG(mesh)
    assert llg.mesh.f_scale_factor == PICOMETERS

def test_can_return_scaled_coordinates():
    mesh = df.Interval(x_n, x_min, x_max)
    monkey_patch_mesh(mesh, NANOMETERS)

    expected_coordinates = np.linspace(x_min, x_max, x_n+1).reshape((x_n+1, -1))
    expected_scaled_coords = expected_coordinates * NANOMETERS
    
    print "test_can_return_scaled_coordinates():"
    print "Coordinates:\n", mesh.coordinates()
    print "Scaled coordinates:\n", mesh.f_scaled_coordinates()

    assert np.allclose(mesh.coordinates(), expected_coordinates,
            rtol=TOL, atol=TOL)
    assert np.allclose(mesh.f_scaled_coordinates(), expected_scaled_coords,
            rtol=TOL, atol=TOL)

def test_can_return_scaled_coordinates_3dim():
    mesh = df.Box(x_min,0,0, x_max,1,1, x_n,1,1)
    monkey_patch_mesh(mesh, NANOMETERS)

    expected_scaled_coordinates = mesh.coordinates() * NANOMETERS
    assert np.allclose(mesh.f_scaled_coordinates(), expected_scaled_coordinates,
            rtol=TOL, atol=TOL)
