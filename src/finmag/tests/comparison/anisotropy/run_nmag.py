# One dimensional magnetic system studied using nsim
import math
import numpy as np
import nmag
from nmag import SI
from test_anisotropy_field import m_gen

# Create the material
mat_Py = nmag.MagMaterial(name="Py",
    Ms=SI(0.86e6, "A/m"),
    exchange_coupling=SI(13.0e-12, "J/m"),
    llg_gamma_G=SI(0.2211e6, "m/A s"),
    anisotropy=nmag.uniaxial_anisotropy(axis=[1, 0, 0], K1=SI(520e3, "J/m^3")),
    llg_damping=SI(0.2),
    llg_normalisationfactor=SI(0.001e12, "1/s"))

# Create the simulation object
sim = nmag.Simulation("1d", do_demag=False)

# Load the mesh
sim.load_mesh("bar_5_5_5.nmesh.h5", [("Py", mat_Py)], unit_length=SI(1e-9, "m"))

# Set the initial magnetisation
sim.set_m(lambda r: m_gen(np.array(r) * 1e9))

# Save the exchange field and the magnetisation once at the beginning
# of the simulation for comparison with finmag
np.savetxt("H_anis_nmag.txt", sim.get_subfield("H_anis_Py"))
np.savetxt("m0_nmag.txt", sim.get_subfield("m_Py"))
