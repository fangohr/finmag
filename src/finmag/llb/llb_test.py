import os
import dolfin as df
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from finmag.llb.llb import LLB
from finmag.llb.exchange import Exchange
from finmag.energies import Zeeman
from finmag.energies import Demag
from finmag.llb.material import Material

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def test_llb_sundials(do_plot=False):
    mesh = df.BoxMesh(0, 0, 0, 2, 2, 2, 1, 1, 1)
    
    mat = Material(mesh, name='FePt',unit_length=1e-9)
    mat.set_m((1, 0, 0))
    mat.T = 10
    mat.alpha=0.1
    
    sim = LLB(mat)
    sim.set_up_solver()
    
    H0 = 1e5
    sim.add(Zeeman((0, 0, H0)))

    dt = 1e-12; ts = np.linspace(0, 1000 * dt, 101)

    precession_coeff = sim.gamma_LL
    mz_ref = []
    
    mxyz = []

    mz = []
    real_ts=[]
    for t in ts:
        sim.run_until(t)
        real_ts.append(sim.t)
        mz_ref.append(np.tanh(precession_coeff * mat.alpha * H0 * sim.t))
        mz.append(sim.m[-1]) # same as m_average for this macrospin problem
        
        sim.m.shape=(3,-1)
        mxyz.append(sim.m[:,-1].copy())
        sim.m.shape=(-1,)
    
    mxyz=np.array(mxyz)

    mz=np.array(mz)
    
    print np.sum(mxyz**2,axis=1)-1
    


    if do_plot:
        ts_ns = np.array(real_ts) * 1e9
        plt.plot(ts_ns, mz, "b.", label="computed") 
        plt.plot(ts_ns, mz_ref, "r-", label="analytical") 
        plt.xlabel("time (ns)")
        plt.ylabel("mz")
        plt.title("integrating a macrospin")
        plt.legend()
        plt.savefig(os.path.join(MODULE_DIR, "test_llb.png"))
    

    print("Deviation = {}".format(np.max(np.abs(mz - mz_ref))))

    #assert np.max(np.abs(mz - mz_ref)) < 1e-7
    

def sim_llb_100(do_plot=False):
    mesh = df.BoxMesh(0, 0, 0, 10, 10, 10, 1, 1, 1)
    
    mat = Material(mesh, name='FePt',unit_length=1e-9)
    mat.set_m((1, 0, 0))
    mat.T =100
    mat.alpha=0.1
    
    print mat.Ms0
    print mat.volumes
    print mat.mat.chi_par(100)
    sim = LLB(mat)
    sim.set_up_stochastic_solver(using_type_II=True)
    
    H0 = 1e5
    sim.add(Zeeman((0, 0, H0)))

    dt = 1e-12; ts = np.linspace(0, 100 * dt, 101)

    precession_coeff = sim.gamma_LL
    mz_ref = []
    
    mz = []
    real_ts=[]
    for t in ts:
        sim.run_until(t)
        real_ts.append(sim.t)
        mz_ref.append(np.tanh(precession_coeff * mat.alpha * H0 * sim.t))
        mz.append(sim.m_average) # same as m_average for this macrospin problem
    
    mz=np.array(mz)
    print mz

    if do_plot:
        ts_ns = np.array(real_ts) * 1e9
        plt.plot(ts_ns, mz, "b.", label="computed") 
        plt.plot(ts_ns, mz_ref, "r-", label="analytical") 
        plt.xlabel("time (ns)")
        plt.ylabel("mz")
        plt.title("integrating a macrospin")
        plt.legend()
        plt.savefig(os.path.join(MODULE_DIR, "test_llb_100K.png"))


def test_llb_save_data():
    mesh = df.BoxMesh(0, 0, 0, 10, 10, 5, 2, 2, 1)
    
    def region1(coords):
        if coords[2]<0.5:
            return True
        else:
            return False
    
    def region2(coords):
        return not region1(coords)
        
    def init_Ms(coords):
        if region1(coords)>0:
            return 8.6e5
        else:
            return 8.0e5
        
    def init_T(pos):
        return 1*pos[2]
    
    mat = Material(mesh, name='FePt',unit_length=1e-9)
    mat.Ms=init_Ms
    mat.set_m((1, 0, 0))
    mat.T = init_T
    mat.alpha=0.1
    
    
    assert(mat.T[0]==0)
    
    sim = LLB(mat,name='test_llb')
    sim.set_up_solver()
    
    ts = np.linspace(0, 1e-11, 11)
    
    H0 = 1e6
    sim.add(Zeeman((0, 0, H0)))
    sim.add(Exchange(mat))
    
    demag=Demag(solver='FK')
    sim.add(demag)
    
    sim.save_m_in_region(region1,name='bottom')
    sim.save_m_in_region(region2,name='top')
    sim.schedule('save_ndt',every=1e-12)
    
    
    for t in ts:
        print 't===',t
        sim.run_until(t)
        

def llb_relax():
    mesh = df.BoxMesh(0, 0, 0, 10, 10, 5, 2, 2, 1)
    
    mat = Material(mesh, name='FePt',unit_length=1e-9)
    mat.set_m((1, 0, 0))
    mat.T = 0
    mat.alpha=0.1
    
    
    assert(mat.T[0]==0)
    
    sim = LLB(mat,name='llb_relax')
    sim.set_up_solver()
        
    H0 = 1e6
    sim.add(Zeeman((0, 0, H0)))
    sim.add(Exchange(mat))
    
    demag=Demag(solver='FK')
    sim.add(demag)
    
    sim.schedule('save_vtk',at_end=True,filename='p0.pvd')
    sim.schedule('save_ndt',at_end=True)
    
    sim.relax()
    

if __name__ == "__main__":
    test_llb_sundials(do_plot=True)
    #sim_llb_100(do_plot=True)
    #test_llb_save_data()
    #llb_relax()


