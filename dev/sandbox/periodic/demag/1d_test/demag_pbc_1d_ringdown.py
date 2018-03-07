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

mesh = df.BoxMesh(-5, -5, -5, 5, 5, 5, 5, 5, 5)



def ring_down(m0=(1,0.01,0), pbc=None):
    
    n = 49
    
    assert n>=1 and n%2==1
    
    Ms = 8.6e5
    
    sim = Simulation(mesh, Ms, unit_length=1e-9, name = 'dy', pbc='1d')
    sim.alpha = 1e-3
    
    sim.set_m(m0)
    sim.set_tol(1e-8, 1e-8)
    
    parameters = {
            'absolute_tolerance': 1e-10,
            'relative_tolerance': 1e-10,
            'maximum_iterations': int(1e5)
    }
    
    Ts = []
    for i in range(-n/2+1,n/2+1):
        Ts.append((10.*i,0,0))
    
    demag = Demag(Ts=Ts)
    
    demag.parameters['phi_1'] = parameters
    demag.parameters['phi_2'] = parameters
    
    sim.add(demag)
    sim.add(Exchange(1.3e-11))
    
    sim.schedule('save_ndt', every=2e-12)
    
    sim.run_until(4e-9)
    


def plot_mx(filename='dy.ndt'):
    
    fig=plt.figure()
    plt.plot(ts, data['E_total'], label='Total')
    #plt.plot(ts, data['E_Demag'], label='Demag')
    #plt.plot(ts, data['E_Exchange'], label='Exchange')
    plt.xlabel('time (ns)')
    plt.ylabel('energy')
    
    plt.legend()
    
    fig.savefig('energy.pdf')

def fft(mx, dt=5e-12):
    n = len(mx)
    freq = np.fft.fftfreq(n, dt)
    
    ft_mx = np.fft.fft(mx)
    
    ft_abs = np.abs(ft_mx)
    ft_phase = np.angle(ft_mx)
    
    return freq, ft_abs, ft_phase

def plot_average_fft():
    
    data = Tablereader('dy.ndt')
    
    ts = data['time']
    my = data['m_y']

    dt = ts[1]-ts[0]
    
    freq, ft_abs, phase = fft(my, dt)
    
    fig=plt.figure()
    plt.subplot(2,1,1)
    plt.plot(ts*1e9,my,label='Real')
    plt.xlabel('Time (ns)')
    plt.ylabel('m_y')
    
    plt.subplot(2,1,2)
    plt.plot(freq*1e-9,ft_abs,'.-',label='Real')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('FFT')
    plt.xlim([10,20])
    #plt.ylim([0,10])
    
    fig.savefig('average_fft.pdf')
    

    
    


if __name__ == '__main__':
    #ring_down()
    plot_average_fft()
