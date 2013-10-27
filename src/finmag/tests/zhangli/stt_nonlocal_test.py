import os
import dolfin as df
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


from finmag import Simulation as Sim
from finmag.energies import Exchange, UniaxialAnisotropy

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def sech(x):
    return 1/np.cosh(x)

def init_m(pos):
    x=pos[0]

    delta = np.sqrt(13e-12/520e3)*1e9
    sx = -np.tanh((x-50)/delta)
    sy = sech((x-50)/delta)
    return (sx,sy,0)

def init_J(pos):
    
    return (1e12,0,0)

def test_zhangli():
    
    #mesh = df.BoxMesh(0, 0, 0, 100, 1, 1, 50, 1, 1)
    mesh = df.IntervalMesh(50,0,100)
    
    sim = Sim(mesh, Ms=8.6e5, unit_length=1e-9, kernel='llg_stt')
    sim.set_m(init_m)
    
    sim.add(UniaxialAnisotropy(K1=520e3, axis=[1, 0, 0]))
    sim.add(Exchange(A=13e-12))
    sim.alpha = 0.01
    
    sim.llg.set_parameters(J_profile=init_J)
    
    p0=sim.m_average
    
    sim.run_until(5e-12)
    p1=sim.m_average

    assert p1[0] < p0[0]
    assert abs(p0[0])<1e-15
    assert abs(p1[0])>1e-3

if __name__ == "__main__":
    test_zhangli()
   


