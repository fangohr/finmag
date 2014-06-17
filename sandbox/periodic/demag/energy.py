import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import dolfin as df
from finmag import Simulation
from finmag.energies import Exchange, DMI, Demag, Zeeman, UniaxialAnisotropy
from finmag.energies.zeeman import TimeZeemanPython
import matplotlib.pyplot as plt
from finmag.util.fileio import Tablereader

mesh = df.BoxMesh(0, 0, 0, 10, 10, 10, 2, 2, 2)
#mesh = df.BoxMesh(0, 0, 0, 30, 30, 100, 6, 6, 20)
#df.plot(mesh, interactive=True)
def relax_system():
    Ms = 8.6e5
    sim = Simulation(mesh, Ms, unit_length=1e-9, name = 'dy', pbc='1d')
    
    sim.alpha = 0.01
    sim.set_m((0.8,0.6,1))
    
    sim.set_tol(1e-8, 1e-8)

    A = 1.3e-11
    
    sim.add(Exchange(A))
    
    parameters = {
            'absolute_tolerance': 1e-10,
            'relative_tolerance': 1e-10,
            'maximum_iterations': int(1e5)
    }
    
    Ts = []
    for i in range(-9,10):
        Ts.append((10*i,0,0))
    
    demag = Demag(solver='Treecode')
    
    demag.parameters['phi_1'] = parameters
    demag.parameters['phi_2'] = parameters
    
    
    sim.add(demag)
    
    demag.compute_field()
    
    sim.schedule('save_ndt', every=2e-12)
    #sim.schedule('save_vtk', every=2e-12, filename='vtks/m.pvd')
    #sim.schedule('save_m', every=2e-12, filename='npys/m.pvd')
    
    sim.run_until(0.2e-9)


def plot_mx(filename='dy.ndt'):
    
    data = Tablereader(filename)
    
    ts=data['time']/1e-9
    
    fig=plt.figure()
    plt.plot(ts, data['E_total'], label='Total')
    #plt.plot(ts, data['E_Demag'], label='Demag')
    #plt.plot(ts, data['E_Exchange'], label='Exchange')
    plt.xlabel('time (ns)')
    plt.ylabel('energy')
    
    plt.legend()
    
    fig.savefig('energy.pdf')
    
    


if __name__ == '__main__':
    #relax()
    relax_system()
    plot_mx()
    
