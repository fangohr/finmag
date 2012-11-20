import os
import dolfin as df
from finmag.util.meshes import from_geofile
from finmag.util.helpers import piecewise_on_subdomains

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
mesh = from_geofile(os.path.join(MODULE_DIR, "cube.geo"))

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
    mesh = from_geofile("cube.geo")
    # This holds the region number for each cell.
    mesh_function = df.MeshFunction("uint", mesh, "cube_mat.xml")
    print "TODO: Change the .geo file so that we define a sphere instead of a cube."

    # XXX Plot the regions for debugging
    print "TODO: Are the regions correctly assigned? Plotting this seems to indicate that region 1 has some stray points around the edge of the mesh"
    import matplotlib.pyplot as plt
    coords = mesh.coordinates()
    region1 = [i for i in xrange(len(coords)) if mesh_function.array()[i] == 1]
    region2 = [i for i in xrange(len(coords)) if mesh_function.array()[i] == 2]
    pts1 = [coords[i] for i in region1]
    pts2 = [coords[i] for i in region2]
    ax = plt.gca(projection='3d')
    ax.scatter3D(*zip(*pts1), color="green")
    #ax.scatter3D(*zip(*pts2), color="red")
    plt.show()

    # Define different values for the saturation magnetisation on each subdomain
    Ms_vals = {
        1: 1e6,  # region 1 = Permalloy
        2: 0,    # region 2 = "air"
        }
    Ms = piecewise_on_subdomains(mesh, mesh_function, Ms_vals)
