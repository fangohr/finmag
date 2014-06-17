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



def compute_field(n=1, m0=(1,0,0), pbc=None):
    
    assert n>=1 and n%2==1
    
    Ms = 1e6
    sim = Simulation(mesh, Ms, unit_length=1e-9, name = 'dy', pbc=pbc)
    
    sim.set_m(m0)
    
    parameters = {
            'absolute_tolerance': 1e-10,
            'relative_tolerance': 1e-10,
            'maximum_iterations': int(1e5)
    }
    
    Ts = []
    for i in range(-n/2+1,n/2+1):
        Ts.append((10.00001*i,0,0))
    
    demag = Demag(Ts=Ts)
    
    demag.parameters['phi_1'] = parameters
    demag.parameters['phi_2'] = parameters
    
    sim.add(demag)
    
    field = sim.llg.effective_field.get_dolfin_function('Demag')
    
    return field(0,0,0)


def plot_mx():
    
    ns = [1,3,5,7,9,11]
    field=[]
    field_pbc=[]
    for n in ns:
        f=compute_field(n=n,m0=(1,0,0))
        field.append(abs(f[0]))
        
        #f2 = compute_field(n=n,m0=(1,0,0),pbc='1d')
        #field_pbc.append(abs(f2[0]))
        
    fig=plt.figure()
    plt.plot(ns, field, '.-',label='field')
    #plt.plot(ns, field_pbc, '.-',label='field2')
    #plt.plot(ts, data['E_Demag'], label='Demag')
    #plt.plot(ts, data['E_Exchange'], label='Exchange')
    plt.xlabel('copies')
    plt.ylabel('field')
    
    plt.legend()
    
    fig.savefig('field_100.pdf')
    
def plot_mx_2():
    
    ns = [1,3,5,7,9,11]
    field=[]
    field_pbc=[]
    for n in ns:
        f=compute_field(n=n,m0=(0,0,1))
        field.append(abs(f[2]))
        
        f2 = compute_field(n=n,m0=(0,0,1),pbc='1d')
        field_pbc.append(abs(f2[2]))
        
    fig=plt.figure()
    plt.plot(ns, field, '.-',label='field')
    plt.plot(ns, field_pbc, '.-',label='field2')
    #plt.plot(ts, data['E_Demag'], label='Demag')
    #plt.plot(ts, data['E_Exchange'], label='Exchange')
    plt.xlabel('copies')
    plt.ylabel('field')
    
    plt.legend()
    
    fig.savefig('field_001.pdf')
    
    


if __name__ == '__main__':
    #relax()
    #relax_system()
    #plot_mx()
    compute_field()
    plot_mx()
    #plot_mx_2()