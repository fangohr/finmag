import numpy as np
import dolfin as df
from finmag import Simulation
from finmag.energies import Exchange, DMI, Zeeman
from dmi_helper import compute_skyrmion_number_2d

def init_skx_down(pos):
    x = pos[0]
    y = pos[1]
    
    if (x-50)**2+(y-50)**2<15**2:
        return (0,0,-1)
    else:
        return (0,0,1)


def compute_skyrmion_number_2d_example():

    mesh = df.CircleMesh(df.Point(0,0),20,4)
    
    Ms = 3.84e5
    mu0 = 4*np.pi*1e-7
    Hz = 0.2
    
    sim = Simulation(mesh, Ms, unit_length=1e-9, name='sim')
    sim.llg.do_precession = False
    
    sim.set_m(init_skx_down)
    
    sim.add(Exchange(8.78e-12))
    sim.add(DMI(-1.58e-3))
    sim.add(Zeeman((0,0,Hz/mu0)))
    
    sim.relax(stopping_dmdt=1, dt_limit=1e-9)
    
    df.plot(sim.llg._m)
    df.interactive()
    
    print compute_skyrmion_number_2d(sim.llg._m)
    
def test_compute_skyrmion_number_2d_pbc():

    mesh = df.RectangleMesh(0,0,100,100,40,40)
    
    Ms = 8.6e5
    sim = Simulation(mesh, Ms, pbc='2d', unit_length=1e-9)
    sim.set_m(init_skx_down)

    sim.add(Exchange(1.3e-11))
    sim.add(DMI(D = 4e-3, interfacial=False))
    sim.add(Zeeman((0,0,0.45*Ms)))
    
    sim.llg.do_precession = False
        
    sim.relax(stopping_dmdt=1, dt_limit=1e-9)
    
    #df.plot(sim.llg._m)
    #df.interactive()
    
    sky_num = compute_skyrmion_number_2d(sim.llg._m)
    
    print 'sky_num = %g'%sky_num
    
    assert sky_num < -0.95 and sky_num > -1.0

if __name__ == "__main__":
    compute_skyrmion_number_2d_example()
    test_compute_skyrmion_number_2d_pbc()