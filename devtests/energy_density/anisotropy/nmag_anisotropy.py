import numpy as np
import nmag
from nmag import SI
import nmeshlib.unidmesher as unidmesher

mesh_unit = SI(1e-9, "m")   # mesh unit (1 nm)
layers = [(0.0, 1.0)]       # the mesh
discretization = 0.2        # discretization

def m0(r):
    """Initial magnetisation 45 degrees between x- and z-axis."""
    return [1/np.sqrt(2), 0, 1/np.sqrt(2)]

mat_Py = nmag.MagMaterial(name="Py",
                          Ms=SI(1,"A/m"),
                          anisotropy=nmag.uniaxial_anisotropy(axis=[0, 0, 1], K1=SI(1, "J/m^3")))

sim = nmag.Simulation("Simple anisotropy", do_demag=False)

# Write mesh to file
mesh_file_name = '1d_x6.nmesh'
mesh_lists = unidmesher.mesh_1d(layers, discretization)
unidmesher.write_mesh(mesh_lists, out=mesh_file_name)

sim.load_mesh(mesh_file_name,
              [("Py", mat_Py)],
              unit_length=mesh_unit)
sim.set_m(m0)

print sim.get_subfield("E_anis_Py")
