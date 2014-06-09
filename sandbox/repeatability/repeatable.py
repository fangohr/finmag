import os
import dolfin as df
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import finmag.util.consts as consts

from sundials_ode import Test_Sundials


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

alpha = 0.01
H0 = 1e5
Ms = 8.6e5

def call_back(t,y):
    
    y.shape = (3,-1)
    
    H = np.zeros(y.shape,dtype=np.double)
    H[2,:] = H0
    
    gamma = consts.gamma
    
    alpha_L = alpha/(1 + alpha ** 2)
    
    dm1 = -gamma * np.cross(y, H, axis=0)
    
    dm2 = alpha_L * np.cross(y, dm1, axis=0)
    
    c = 1e11
    
    mm = y[0]*y[0] + y[1]*y[1] + y[2]*y[2]
    dm3 = c*(1-mm)*y
    
    dm = dm1+dm2+dm3
    
    y.shape=(-1,)
    dm.shape=(-1,)
    
    return dm


def test_sim_ode(do_plot=False):
    
    m0 = np.array([1.0,1.0,0,0,0,0])
    
    sim = Test_Sundials(call_back, m0)
    sim.create_integrator(reltol=1e-8, abstol=1e-8)
    
    
    ts = np.linspace(0, 5e-9, 101)

    precession_coeff = consts.gamma / (1 + alpha ** 2)
    mz_ref = np.tanh(precession_coeff * alpha * H0 * ts)

    mzs = []
    length_error = []
    for t in ts:
        sim.advance_time(t)
        mm = sim.x.copy()
        print mm

        mm.shape=(3,-1)
        mx,my,mz = mm[:,-1] # same as m_average for this macrospin problem
        mm.shape=(-1,)
        mzs.append(mz)

        length = np.sqrt(mx**2+my**2+mz**2)
        
        length_error.append(abs(length-1.0))

    if do_plot:
        ts_ns = ts * 1e9
        plt.plot(ts_ns, mzs, "b.", label="computed")
        plt.plot(ts_ns, mz_ref, "r-", label="analytical")
        plt.xlabel("time (ns)")
        plt.ylabel("mz")
        plt.title("integrating a macrospin")
        plt.legend()
        plt.savefig(os.path.join(MODULE_DIR, "test_sim_ode.png"))

    print("max deviation = {}, last mz={} and last length error={}".format(
            np.max(np.abs(mzs - mz_ref)),
            mzs[-1],length_error[-1])
          )

if __name__ == "__main__":
    test_sim_ode(do_plot=True)
