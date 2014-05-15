import os
import dolfin as df
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


from finmag import Simulation as Sim
from finmag.energies import Exchange, UniaxialAnisotropy

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def test_compute_dm():
    from finmag.physics.neb import compute_dm
    
    a1 = np.array([0,0,1])
    a2 = np.array([0,1,0])
    
    assert(compute_dm(a1,a2)**2-2.0)<1e-15
    
    
def test_cartesian2spherical_field():
    from finmag.physics.neb import cartesian2spherical_field
    
    Kx=1e5
    Kp=6e4
    
    theta = 0.17
    phi = 0.23
    
    theta_phi=np.array([theta,phi])
    
    mx = np.sin(theta)*np.cos(phi)
    my = np.sin(theta)*np.sin(phi)
    mz = np.cos(theta)
    
    E = -Kx*mx**2 + Kp*mz**2
    
    hx = -2*Kx*mx
    hz = 2*Kp*mz
    H = np.array([hx,0,hz])
    
    pE_theta = -2*Kx*mx*np.cos(phi)*np.cos(theta) + 2*Kp*mz*(-np.sin(theta))
    pE_phi = 2*Kx*mx*np.sin(theta)*np.sin(phi)
    
    res = cartesian2spherical_field(H,theta_phi)
    assert abs(res[0]-pE_theta)<1e-15
    assert abs(res[1]-pE_phi)<1e-12

if __name__ == "__main__":
    
    test_compute_dm()
    
    test_cartesian2spherical_field()


