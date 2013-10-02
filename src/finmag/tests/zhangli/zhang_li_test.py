import os
import dolfin as df
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


from finmag import Simulation as Sim
from finmag.energies import Exchange, UniaxialAnisotropy

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def init_m(pos):
    x=pos[0]

    delta = np.sqrt(13e-12/520e3)*1e9
    sy = 1/np.cosh((x-50)/delta)
    sx = -np.sinh((x-50)/delta)
    return (sx,sy,0)

def init_J(pos):
    
    return (1e12,0,0)

def test_zhangli():
    
    mesh = df.BoxMesh(0, 0, 0, 100, 1, 1, 50, 1, 1)
    
    sim = Sim(mesh, Ms=8.6e5, unit_length=1e-9)
    sim.set_m(init_m)
    
    sim.add(UniaxialAnisotropy(K1=520e3, axis=[1, 0, 0]))
    sim.add(Exchange(A=13e-12))
    sim.alpha = 0.01
    
    sim.set_zhangli(init_J, 0.5,0.01)
    
    p0=sim.m_average
    
    sim.run_until(1e-11)
    p1=sim.m_average

    assert p1[0] < p0[0]
    assert abs(p0[0])<1e-15
    assert abs(p1[0])>1e-3
    

if __name__ == "__main__":
    test_zhangli()


