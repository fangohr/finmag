import os
import dolfin as df
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from finmag.llb.llb import LLB
from finmag.energies import Zeeman
from finmag.energies import Demag
from finmag.llb.material import Material

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def test_llb_sundials(do_plot=False):
    mesh = df.BoxMesh(0, 0, 0, 2, 2, 2, 1, 1, 1)
    
    mat = Material(mesh, name='FePt',unit_length=1e-9)
    mat.set_m((1, 0, 0))
    mat.T = 0
    mat.alpha=0.1
    
    sim = LLB(mat)
    sim.set_up_solver()
    
    H0 = 1e5
    sim.add(Zeeman((0, 0, H0)))

    dt = 1e-12; ts = np.linspace(0, 500 * dt, 101)

    precession_coeff = mat.gamma_LL
    mz_ref = []
    
    mz = []
    real_ts=[]
    for t in ts:
        sim.run_until(t)
        real_ts.append(sim.t)
        mz_ref.append(np.tanh(precession_coeff * mat.alpha * H0 * sim.t))
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
        plt.savefig(os.path.join(MODULE_DIR, "test_llb.png"))

    print("Deviation = {}, total value={}".format(
            np.max(np.abs(mz - mz_ref)),
            mz_ref))
    #TODO: debug why the deviation is so large
    assert np.max(np.abs(mz - mz_ref)) < 4e-2
    

if __name__ == "__main__":
    test_llb_sundials(do_plot=True)
    


