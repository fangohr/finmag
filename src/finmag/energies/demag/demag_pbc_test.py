import numpy as np
import dolfin as df
from finmag import Simulation
from finmag.energies import Exchange, DMI, Demag
from finmag import MacroGeometry

mesh_1 = df.BoxMesh(-10, -10, -10, 10, 10, 10, 10, 10, 10)
mesh_3 = df.BoxMesh(-30, -10, -10, 30, 10, 10, 30, 10, 10)
mesh_9 = df.BoxMesh(-30, -30, -10, 30, 30, 10, 30, 30, 10)

def compute_field(mesh, nx=1, ny=1, m0=(1,0,0), pbc=None):
        
    Ms = 1e6
    sim = Simulation(mesh, Ms, unit_length=1e-9, name = 'dy', pbc=pbc)
    
    sim.set_m(m0)
    
    parameters = {
            'absolute_tolerance': 1e-10,
            'relative_tolerance': 1e-10,
            'maximum_iterations': int(1e5)
    }
    
    demag = Demag(macrogeometry=MacroGeometry(nx=nx,ny=ny))
    
    demag.parameters['phi_1'] = parameters
    demag.parameters['phi_2'] = parameters
    
    sim.add(demag)
    
    field = sim.llg.effective_field.get_dolfin_function('Demag')
    
    return field(0,0,0)/Ms


def test_field_1d():
    m0 = (1,0,0)
    f1 = compute_field(mesh_1,nx=3,m0=m0)
    f2 = compute_field(mesh_3,nx=1,m0=m0)
    error = abs((f1 - f2)/f2)
    print  f1,f2,error
    assert max(error)<0.012

    m0 = (0,0,1)
    f1 = compute_field(mesh_1,nx=3,m0=m0)
    f2 = compute_field(mesh_3,nx=1,m0=m0)
    error = abs((f1 - f2)/f2)
    print  f1,f2,error
    assert max(error)<0.02


def test_field_2d():
    m0 = (1,0,0)
    f1 = compute_field(mesh_1,nx=3,ny=3,m0=m0)
    f2 = compute_field(mesh_9,m0=m0)
    error = abs((f1 - f2)/f2)
    print  f1,f2,error
    assert max(error)<0.01

    m0 = (0,0,1)
    f1 = compute_field(mesh_1,nx=3,ny=3,m0=m0)
    f2 = compute_field(mesh_9,m0=m0)
    error = abs((f1 - f2)/f2)
    print  f1,f2,error
    assert max(error)<0.004

if __name__ == '__main__':
    test_field_1d()
    test_field_2d()
