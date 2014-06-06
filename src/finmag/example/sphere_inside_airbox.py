from finmag.sim.sim import sim_with
from finmag.util.meshes import sphere_inside_box
from finmag.util.helpers import scalar_valued_dg_function
import numpy as np


def sphere_inside_airbox(r_sphere=5.0, r_shell=10.0, l_box=50.0, maxh_sphere=2.0, maxh_shell=None, maxh_box=10.0, center_sphere=[0, 0, 0], m_init=[1, 0, 0], Ms=8.6e5, A=13e-12, **kwargs):
    """
    Create a Simulation object of a sphere inside a box.

    The saturation magnetisation of the region outside the
    sphere is set to a very low value so that it effectively
    appears to be non-magnetic ('air'). However, the demag
    field can be sampled in this region and represents the
    stray field of the sphere.

    *Arguments*

    r_shere:

        Radius of the sphere.

    r_shell:

        Radius of the 'shell' enclosing the sphere. The region
        between the sphere and the shell will not be meshed.
        Everything between the spherical shell and the outer
        edges of the box is considered 'air'.

    l_box:

        The edge length of the box.

    maxh_sphere:

        Mesh discretization of the sphere.

    maxh_box:

        Mesh discretization of the box (i.e. the 'air' region).

    maxh_shell:

        This value determines how finely the inner border of the
        'air' region which surrounds the sphere is discretized.
        Default: same as the enclosed sphere.

    center_sphere:

        Center of the inner sphere (default: (0, 0, 0). The box
        is always centered at the origin, so this can be used to
        shift the sphere relative to the box.

    m_init:

        Initial magnetisation of the sphere (is normalised
        automatically).

    Ms:

        Saturation magnetisation of the sphere.

    A:

        Exchange coupling constant of the sphere.

    All remaining keyword arguments are passed on to the `sim_with`
    command which is used to create the Simulation object.


    *Returns*

    A Simulation object representing the sphere inside the airbox.

    """
    mesh = sphere_inside_box(r_sphere=r_sphere, r_shell=r_shell, l_box=l_box,
                             maxh_sphere=maxh_sphere, maxh_box=maxh_box,
                             maxh_shell=maxh_shell, center_sphere=center_sphere)

    ## Create a Simulation object using this mesh with a tiny Ms in the "air" region
    def Ms_pyfun(pt):
        if np.linalg.norm(pt) <= 0.5 * (r_sphere + r_shell):
            return Ms
        else:
            return 1.
    Ms_fun = scalar_valued_dg_function(Ms_pyfun, mesh)

    def fun_region_marker(pt):
        if np.linalg.norm(pt - center_sphere) <= 0.5 * (r_sphere + r_shell):
            return "sphere"
        else:
            return "air"

    sim = sim_with(mesh, Ms_fun, m_init=m_init, A=A, unit_length=1e-9, name='sphere_inside_airbox', **kwargs)
    sim.mark_regions(fun_region_marker)

    return sim
