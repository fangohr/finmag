import os
import numpy as np
from finmag.util.oommf.ovf import OVFFile

MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"
EPSILON = 1e-15

bar_size = (30e-9, 30e-9, 100e-9)
bar_x_mid = bar_size[0]/2; bar_y_mid = bar_size[1]/2; bar_z_mid = bar_size[2]/2;

def extract_line_across_scalar_field(ovf_filename, txt_filename):
    ovf_file = OVFFile(ovf_filename)
    fl = ovf_file.get_field()

    bar_center_idx = fl.lattice.get_closest((bar_x_mid, bar_y_mid, bar_z_mid))
    bar_center_coords = fl.lattice.get_pos_from_idx(bar_center_idx)
    print "In {}:".format(ovf_filename)
    print "  The node closest to the center has the indices {} and coordinates\n  {}.".format(
            bar_center_idx, bar_center_coords)

    assert abs(bar_center_coords[0] - bar_x_mid) < EPSILON
    assert abs(bar_center_coords[1] - bar_y_mid) < EPSILON

    bar_center_x, bar_center_y, _ = bar_center_idx
    energies_along_z_direction = fl.field_data[0][bar_center_x][bar_center_y]
    coords_of_z_axis = [fl.lattice.get_pos_from_idx((bar_center_x, bar_center_y, i))[2] for i in range(50)]
    np.savetxt(MODULE_DIR + txt_filename, energies_along_z_direction)
    np.savetxt(MODULE_DIR + "oommf_coords_z_axis.txt", coords_of_z_axis)

extract_line_across_scalar_field(
    "bar-Oxs_Demag-demag-Energy_density-00-0000760.oef",
    "oommf_demag_Edensity.txt")

extract_line_across_scalar_field(
    "bar-Oxs_UniformExchange-exc-Energy_density-00-0000760.oef",
    "oommf_exch_Edensity.txt")
