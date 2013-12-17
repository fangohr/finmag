import numpy as np
import dolfin as df
from finmag.util.helpers import stats
from finmag.energies import UniaxialAnisotropy, CubicAnisotropy
from finmag import Simulation

Ms = 8.6e5; K1 = 520e3;
mx = 0.8; my=0.6; mz=0
mu0 = 4*np.pi*1e-7

def test_anisotropy():
    mesh = df.IntervalMesh(1,0,1)
    sim = Simulation(mesh, Ms, unit_length=1e-9)
    sim.set_m((mx,my,mz))
    
    sim.add(UniaxialAnisotropy(K1, axis=[1,0,0]))
    
    expected = 2*K1/(mu0*Ms)*mx
    field = sim.effective_field()
    assert abs(field[0]-expected)/Ms < 1e-15

def m_init(pos):
    x = pos[0]
    return (1,1,0)

def cubic_anisotropy(K1=520e3, K2=0, K3=0):
    mesh = df.RectangleMesh(0,0,50,2,20,1)
    
    sim = Simulation(mesh, Ms, unit_length=1e-9)
    sim.set_m(m_init)
    
    sim.add(CubicAnisotropy(K1=K1, K2=K2, K3=K3, u1=(1,0,0), u2=(0,1,0),assemble=False))

    field1 = sim.effective_field()
    

if __name__ == "__main__":
    test_anisotropy()
    cubic_anisotropy()
