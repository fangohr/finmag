import numpy as np
import nmag, os
from nmag import SI
import nmeshlib.unidmesher as unidmesher


def main():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    L = 10
    mesh_unit = SI(1e-9, "m")   # mesh unit (1 nm)
    layers = [(0.0, L)]         # the mesh
    discretization = 0.1        # discretization

    # Initial magnetization
    xfactor = float(SI("m")/(L*mesh_unit))

    def m0(r):
        return [np.cos(r[0]*np.pi*xfactor), np.sin(r[0]*np.pi*xfactor), 0]

    mat_Py = nmag.MagMaterial(name="Py",
                              Ms=SI(1,"A/m"))

    sim = nmag.Simulation("Hans' configuration", do_demag=False)

    mesh_file_name = '1d.nmesh'
    mesh_lists = unidmesher.mesh_1d(layers, discretization)
    unidmesher.write_mesh(mesh_lists, out=mesh_file_name)

    sim.load_mesh(mesh_file_name,
                  [("Py", mat_Py)],
                  unit_length=mesh_unit)
    sim.set_m(m0)

    np.save(os.path.join(MODULE_DIR, "nmag_hansconf.npy"), sim.get_subfield("E_exch_Py"))


if __name__ == '__main__':
    main()
