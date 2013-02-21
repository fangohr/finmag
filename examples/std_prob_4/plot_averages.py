import os
import numpy as np
import matplotlib.pyplot as plt

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# plots of average magnetisation components

averages_martinez = os.path.join(MODULE_DIR, "m_averages_ref_martinez.txt")
ref_t, ref_mx, ref_my, ref_mz = np.loadtxt(averages_martinez, unpack=True)
plt.plot(ref_t, ref_mx, "r-", label="$m_\mathrm{x}\,\mathrm{Martinez\, et\, al.}$")
plt.plot(ref_t, ref_my, "r:", label="$m_\mathrm{y}$")
plt.plot(ref_t, ref_mz, "r--", label="$m_\mathrm{z}$")

averages_finmag = os.path.join(MODULE_DIR, "dynamics.ndt")
t, mx, my, mz = np.loadtxt(averages_finmag, unpack=True)
t *= 1e9  # convert from s to ns
plt.plot(t, mx, "b-", label="$m_\mathrm{x}\,\mathrm{FinMag}$")
plt.plot(t, my, "b:")
plt.plot(t, mz, "b--")

plt.xlabel("$\mathrm{time}\, (\mathrm{ns})$")
plt.ylabel("$<m_i> = <M_i>/M_\mathrm{S}$")
plt.legend()
plt.xlim([0, 2])
plt.savefig(os.path.join(MODULE_DIR, "m_averages.pdf"))
