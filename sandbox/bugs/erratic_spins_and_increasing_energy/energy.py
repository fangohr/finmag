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

mesh = df.Mesh('mesh_a3cc0220d03c921e8cfa9ecf5da5fc74.xml.gz')


def relax_system():
    Ms = 8.6e5
    sim = Simulation(mesh, Ms, unit_length=1e-9, name = 'dy')
    
    sim.alpha = 0.001
    sim.set_m(np.load('m_000088.npy'))
    
    #sim.set_tol(1e-6, 1e-6)

    A = 1.3e-11
    
    sim.add(Exchange(A))
    
    
    parameters = {
            'absolute_tolerance': 1e-10,
            'relative_tolerance': 1e-10,
            'maximum_iterations': int(1e5)
    }
    
    demag = Demag()
    
    demag.parameters['phi_1'] = parameters
    demag.parameters['phi_2'] = parameters
    
    print demag.parameters
    
    sim.add(demag)
    
    sim.schedule('save_ndt', every=2e-12)
    sim.schedule('save_vtk', every=2e-12, filename='vtks/m.pvd')
    sim.schedule('save_m', every=2e-12, filename='npys/m.pvd')
    
    sim.run_until(1e-9)


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
    #relax_system()
    plot_mx()
    
