import os
import dolfin as df
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from finmag.energies import Zeeman
from finmag.energies import Demag
from sandbox.baryakhter.baryakhtar import LLB
from sandbox.baryakhter.exchange import Exchange

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def test_llb(do_plot=False):
    #mesh = df.BoxMesh(0, 0, 0, 2, 2, 2, 1, 1, 1)
    mesh = df.IntervalMesh(1,0,2)
    Ms = 8.6e5
    sim = LLB(mesh)
    
    sim.alpha = 0.5
    sim.beta = 0.0
    sim.M0 = Ms
    sim.set_M((Ms,0,0))
    
    sim.set_up_solver(reltol=1e-7, abstol=1e-7)
    
    sim.add(Exchange(1.3e-11,chi=1e-7))
    
    H0 = 1e5
    sim.add(Zeeman((0, 0, H0)))

    steps = 100
    dt = 1e-12; ts = np.linspace(0, steps * dt, steps+1)

    precession_coeff = sim.gamma
    mz_ref = []
    
    mz = []
    real_ts=[]
    for t in ts:
        print t,sim.Ms
        sim.run_until(t)
        real_ts.append(sim.t)
        mz_ref.append(np.tanh(precession_coeff * sim.alpha * H0 * sim.t))
        #mz.append(sim.M[-1]/Ms) # same as m_average for this macrospin problem
        mz.append(sim.m[-1])
    
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
    
    assert np.max(np.abs(mz - mz_ref)) < 2e-7

    

if __name__ == "__main__":
    test_llb(do_plot=True)


