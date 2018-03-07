import dolfin as df
import numpy as np
import pylab as plt
from finmag import Simulation as Sim
from finmag.energies import Exchange, DMI, Zeeman
from finmag.energies.zeeman import TimeZeemanPython
from finmag.util.meshes import nanodisk
from finmag.util.consts import mu0

def skyrm_init(x):
    r = (x[0]**2 + x[1]**2)**0.5
    if r < 50e-9:
        return (0, 0, -1)
    else:
        return (0, 0, 1)


def create_mesh():
    d = 30  # nm
    thickness = 5  # nm
    hmax = 3  # nm
    mesh = nanodisk(d, thickness, hmax, save_result=False)
    #mesh = 
    #df.plot(mesh,interactive=True)
    return mesh


def create_sim(mesh):
    
    Ms = 3.84e5  # A/m
    A = 8.78e-12  # J/m 
    D = 1.58e-3  # J/m**2
    
    sim = Sim(mesh, Ms, unit_length=1e-9)
    sim.set_m((0, 0, 1))
    sim.set_tol(reltol=1e-10, abstol=1e-10)

    sim.add(Exchange(A))
    sim.add(DMI(D))
    
    return sim
    

def relax():
    mesh = create_mesh()
    sim = create_sim(mesh)
    sim.relax(stopping_dmdt=1e-3)
    np.save('m0.npy', sim.m)


def dynamics():
    t_exc = 0.5e-9
    H_exc_max = 5e-3 / mu0
    fc = 20e9

    mesh = create_mesh()
    sim = create_sim(mesh)
    sim.set_m(np.load('m0.npy'))
    sim.alpha = 1e-4
    
    H0 = df.Expression(("0","0","1"))
    def time_fun(t):
        if t<t_exc:
            return H_exc_max*np.sinc(2*np.pi*fc*(t-t_exc/2))
        else:
            return 0
    
    zeeman = TimeZeemanPython(H0,time_fun)
    sim.add(zeeman)
    
    t_end = 5e-9    
    delta_t = 5e-12
    
    t_array = np.arange(0, t_end, delta_t)
    for t in t_array:
        sim.run_until(t)
        df.plot(sim.llg._m)


if __name__=='__main__':
    #relax()
    dynamics()
