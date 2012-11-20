import os
import dolfin as df
import matplotlib.pyplot as plt
from finmag.util.meshes import from_geofile, plot_mesh_regions
from finmag.util.helpers import piecewise_on_subdomains

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def test_airbox_method():
    """
    Define a mesh with two regions (Permalloy and 'air'), where
    the Permalloy region is e.g. a sphere of radius 1.0 and the 'air'
    region is a cube with sides of length 10. Next set Ms in both
    regions, where Ms in the 'air' region is either zero or has a
    very low value.  Then run the simulation and and check that the
    value of the external field in the 'air' region coincides with the
    field of a dipole.
    """
    mesh = from_geofile("mesh.geo")
    mesh_region = df.MeshFunction("uint", mesh, "mesh_mat.xml")

    # Visualise mesh regions to check they look right (this will be
    # removed in the final test).
    plot_mesh_regions(mesh_region, regions=[1, 2], colors=["green", "red"],
                      alphas=[1.0, 0.25])

    # Define different values for the saturation magnetisation on each subdomain
    Ms_vals = (8.6e5, 0)
    Ms = piecewise_on_subdomains(mesh, mesh_region, Ms_vals)

if __name__ == '__main__':
    test_airbox_method()
