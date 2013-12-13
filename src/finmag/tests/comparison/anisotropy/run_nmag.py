import nmag
import numpy as np
from nmag import SI

Ms = 0.86e6; K1 = 520e3; a = (1, 0, 0);
x1 = y1 = z1 = 20; # same as in bar.geo file

def m_gen(r):
    x = np.maximum(np.minimum(r[0]/x1, 1.0), 0.0) # x, y and z as a fraction
    y = np.maximum(np.minimum(r[1]/y1, 1.0), 0.0) # between 0 and 1 in the 
    z = np.maximum(np.minimum(r[2]/z1, 1.0), 0.0)
    mx = (2 - y) * (2 * x - 1) / 4
    mz = (2 - y) * (2 * z - 1) / 4 
    my = np.sqrt(1 - mx**2 - mz**2)
    return np.array([mx, my, mz])

def generate_anisotropy_data(anis,name='anis'):
    # Create the material
    mat_Py = nmag.MagMaterial(name="Py",
                Ms=SI(0.86e6, "A/m"),
                anisotropy=anis)

    # Create the simulation object
    sim = nmag.Simulation("anis", do_demag=False)

    # Load the mesh
    sim.load_mesh("bar.nmesh.h5", [("Py", mat_Py)], unit_length=SI(1e-9, "m"))

    # Set the initial magnetisation
    sim.set_m(lambda r: m_gen(np.array(r) * 1e9))

    # Save the exchange field and the magnetisation once at the beginning
    # of the simulation for comparison with finmag
    np.savetxt("H_%s_nmag.txt"%name, sim.get_subfield("H_anis_Py"))
    np.savetxt("m0_nmag.txt", sim.get_subfield("m_Py"))


if __name__ == "__main__":
    #define uniaxial_anisotropy
    anis=nmag.uniaxial_anisotropy(axis=[1, 0, 0], K1=SI(520e3, "J/m^3"))
    generate_anisotropy_data(anis)
    
    #define cubic_anisotropy
    #doesn't work for nmag2.1 ??
    cubic=nmag.cubic_anisotropy(axis1=[1, 0, 0], axis2=[0, 1, 0], K1=SI(48e3, "J/m^3"))
    #generate_anisotropy_data(cubic,name='cubic_anis')