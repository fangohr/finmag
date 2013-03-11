import os
import dolfin as df
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from finmag.llb.sllg import SLLG
from finmag.energies import Zeeman
from finmag.energies import Demag

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def test_sllg_zero_temperature(do_plot=False):
    mesh = df.BoxMesh(0, 0, 0, 2, 2, 2, 1, 1, 1)
    sim = SLLG(mesh, 8.6e5, unit_length=1e-9)
    alpha=0.1
    sim.alpha = alpha
    sim.set_m((1, 0, 0))
    

    H0 = 1e5
    sim.add(Zeeman((0, 0, H0)))

    dt = 1e-12; ts = np.linspace(0, 500 * dt, 100)

    precession_coeff = sim.gamma / (1 + alpha ** 2)
    mz_ref = []
    
    mz = []
    real_ts=[]
    for t in ts:
        sim.run_until(t)
        real_ts.append(sim.t)
        mz_ref.append(np.tanh(precession_coeff * alpha * H0 * sim.t))
        mz.append(sim.m[-1]) # same as m_average for this macrospin problem
    
    mz=np.array(mz)

    if do_plot:
        ts_ns = np.array(real_ts) * 1e9
        plt.plot(ts_ns, mz, "b.", label="computed") 
        plt.plot(ts_ns, mz_ref, "r-", label="analytical") 
        plt.xlabel("time (ns)")
        plt.ylabel("mz")
        plt.title("integrating a macrospin")
        plt.legend()
        plt.savefig(os.path.join(MODULE_DIR, "test_sllg.png"))

    print("Deviation = {}, total value={}".format(
            np.max(np.abs(mz - mz_ref)),
            mz_ref))
    
    assert np.max(np.abs(mz - mz_ref)) < 8e-7
    
def test_sllg_100(do_plot=False):
    mesh = df.BoxMesh(0, 0, 0, 10, 10, 10, 1, 1, 1)
    sim = SLLG(mesh, 8.6e5, unit_length=1e-9)
    alpha=0.01
    sim.alpha = alpha
    sim.T=100
    sim.dt=1e-13
    sim.set_m((1, 0, 0))
    
    #sim.dt=1e-15
    mu0=4*np.pi*1e-7
    H0 = 1.0/mu0
    sim.add(Zeeman((0, 0, H0)))
    
    dt = 1e-12; ts = np.linspace(0, 500 * dt, 101)

    precession_coeff = sim.gamma / (1 + alpha ** 2)
    mz_ref = []
    
    mz = []
    mz_av=[]
    
    for t in ts:
        sim.run_until(t)
        mz_av.append(sim.m_average[2])
        mz_ref.append(np.tanh(precession_coeff * alpha * H0 * sim.t))
        mz.append(sim.m[-1]) # same as m_average for this macrospin problem
    
    mz=np.array(mz)

    if do_plot:
        ts_ns = np.array(ts) * 1e9
        plt.plot(ts_ns, mz, "b.", label="computed") 
        plt.plot(ts_ns, mz_av, "g-", label="average_z")
        plt.plot(ts_ns, mz_ref, "r-", label="analytical") 
        plt.xlabel("time (ns)")
        plt.ylabel("mz")
        plt.title("integrating a macrospin")
        plt.legend()
        plt.savefig(os.path.join(MODULE_DIR, "test_sllg_T100.png"))

    print("Deviation = {}, total value={}".format(
            np.max(np.abs(mz - mz_ref)),
            mz_ref))
    
    #here the test is meaningless
    assert np.max(np.abs(mz - mz_ref)) < 0.5
    

def test_sllg_time():
    mesh = df.BoxMesh(0, 0, 0, 5, 5, 5, 1, 1, 1)
    sim = SLLG(mesh, 8.6e5, unit_length=1e-9)
    sim.alpha = 0.1
    sim.set_m((1, 0, 0))
    ts = np.linspace(0, 1e-9, 1001)
    
    H0 = 1e5
    sim.add(Zeeman((0, 0, H0)))
    
    real_ts=[]
    for t in ts:
        sim.run_until(t)
        real_ts.append(sim.t)
    
    print("Max Deviation = {}".format(
            np.max(np.abs(ts - real_ts))))
    
    assert np.max(np.abs(ts - real_ts)) < 1e-24
    

def test_sllg_save_data():
    mesh = df.BoxMesh(0, 0, 0, 1, 1, 1, 2, 2, 2)
    
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
            return 4e5
    
    
    sim = SLLG(mesh, init_Ms, unit_length=1e-9)
    sim.alpha = 0.1
    sim.set_m((1, 0, 0))
    ts = np.linspace(0, 1e-10, 101)
    
    H0 = 1e6
    sim.add(Zeeman((0, 0, H0)))
    
    demag=Demag(solver='FK')
    sim.add(demag)
    
    sim.save_m_in_region(region1,name='bottom')
    sim.save_m_in_region(region2,name='top')
    
    for t in ts:
        sim.run_until(t)
        
    
def plot_random_number():
    from finmag.native.llb import RandomMT19937
    mt = RandomMT19937() 
    mt.initial_random(422353390)
    
    x=np.zeros(10000000,dtype=np.float)
    for i in range(100):    
        mt.gaussian_random_np(x)
        if i>80:
            plt.cla()
            plt.hist(x, 100, normed=1, facecolor='green', alpha=0.75)
            plt.grid(True)
            plt.savefig(os.path.join(MODULE_DIR, "test_mt19937_%d.png"%i))
        
        print 'step=',i
        

def plot_random_number_np():
    
    np.random.seed(422353390)
    
    for i in range(100):
        x=np.random.randn(10000000)    
    
        plt.cla()
        plt.hist(x, 100, normed=1, facecolor='green', alpha=0.75)
        plt.grid(True)
        plt.savefig(os.path.join(MODULE_DIR, "test_np_%d.png"%i))
    
    

if __name__ == "__main__":
    #test_sllg_zero_temperature(do_plot=True)
    #test_sllg_100(do_plot=True)
    #test_sllg_time()
    #test_sllg_save_data()
    plot_random_number()
    #plot_random_number_np()
    


