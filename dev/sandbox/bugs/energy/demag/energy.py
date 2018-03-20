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


mesh = df.BoxMesh(-100,0,0,100,40,4,100,20,2)

def m_init_dw(pos):
    
    x = pos[0]
    
    if x < -10:
        return (1,0,0)
    elif  x > 10:
        return (-1,0,0)
    else:
        return (0,1,0)
    
    
def varying_field(pos):
    x = pos[0]
    
    return (1e5,100*x,0)


def excite_system():
    Ms = 8.0e5
    sim = Simulation(mesh, Ms, unit_length=1e-9, name = 'dy')
    
    sim.alpha = 0.001
    #sim.set_m(np.load('relaxed.npy'))
    sim.set_m((1,0,0))
    sim.set_tol(1e-8, 1e-8)

    A = 1.3e-11
    
    Kx = 8e4
    #sim.add(Exchange(A))
    sim.add(Zeeman(varying_field))
    #sim.add(UniaxialAnisotropy(Kx,axis=[1,0,0], name='Kx'))
    
    parameters = {
            'absolute_tolerance': 1e-10,
            'relative_tolerance': 1e-10,
            'maximum_iterations': int(1e5)
    }
    
    demag = Demag()
    
    demag.parameters['phi_1'] = parameters
    demag.parameters['phi_2'] = parameters
    
    sim.add(demag)
    
    sim.schedule('save_ndt', every=2e-12)
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
    excite_system()
    plot_mx()
    
