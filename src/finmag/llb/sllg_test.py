import os
import dolfin as df
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from finmag.llb.sllg import SLLG
from finmag.energies import Zeeman


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def test_sllg_zero_temperature(do_plot=False):
    mesh = df.Box(0, 0, 0, 2, 2, 2, 1, 1, 1)
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
    
    assert np.max(np.abs(mz - mz_ref)) < 1e-8
    
def test_sllg_100(do_plot=False):
    mesh = df.Box(0, 0, 0, 5, 5, 5, 1, 1, 1)
    sim = SLLG(mesh, 8.6e5, unit_length=1e-9)
    alpha=0.1
    sim.alpha = alpha
    sim.dt=5e-15
    sim.set_m((1, 0, 0))
    sim.T=100
    #sim.dt=1e-15
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
        plt.savefig(os.path.join(MODULE_DIR, "test_sllg_T100.png"))

    print("Deviation = {}, total value={}".format(
            np.max(np.abs(mz - mz_ref)),
            mz_ref))
    
    assert np.max(np.abs(mz - mz_ref)) < 0.16
    

def test_sllg_time():
    mesh = df.Box(0, 0, 0, 5, 5, 5, 1, 1, 1)
    sim = SLLG(mesh, 8.6e5, unit_length=1e-9)
    sim.alpha = 0.1
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
    

if __name__ == "__main__":
    test_sllg_zero_temperature(do_plot=True)
    test_sllg_100(do_plot=True)
    test_sllg_time()
    


