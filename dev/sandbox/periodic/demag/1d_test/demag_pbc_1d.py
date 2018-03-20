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

mesh = df.BoxMesh(-20, -20, -20, 20, 20, 20, 20, 20, 20)

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
        Ts.append((40.*i,0,0))
    
    demag = Demag(Ts=Ts)
    
    demag.parameters['phi_1'] = parameters
    demag.parameters['phi_2'] = parameters
    
    sim.add(demag)
    
    sim.set_m((1,0,0))
    field1 = sim.llg.effective_field.get_dolfin_function('Demag')
    
    sim.set_m((0,0,1))
    field2 = sim.llg.effective_field.get_dolfin_function('Demag')
    
    return (field1(0,0,0)/Ms, field2(0,0,0)/Ms)


def plot_field():
    
    ns = [1, 3, 5, 7, 11, 15, 21, 29, 59]
    ns = [1,3,5,7, 29]
    field1=[]
    field2=[]
    for n in ns:
        f, g=compute_field(n=n)
        field1.append(abs(f[0]))
        field2.append(abs(g[2]))
        
        #f2 = compute_field(n=n,m0=(1,0,0),pbc='1d')
        #field_pbc.append(abs(f2[0]))
        
    fig=plt.figure(figsize=(5, 5))
    plt.subplot(2, 1, 1)
    plt.plot(ns, field1, '.-')
    plt.xlabel('Copies')
    plt.ylabel('Field (Ms)')
    plt.title('m aligned along x')
    
    plt.subplot(2, 1, 2)
    plt.plot(ns, field2, '.-')
    
    plt.xlabel('Copies')
    plt.ylabel('Field (Ms)')
    plt.title('m aligned along z')
    #plt.legend()
    
    fig.savefig('fields.pdf')
    

    
    


if __name__ == '__main__':
    plot_field()
