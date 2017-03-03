import os
import sys
import numpy as np

if not os.path.isfile("spinxfer-onespin.odt"):
    print("Please run oommf on the run.mif file.")
    sys.exit(1)

oommf_data = np.loadtxt("spinxfer-onespin.odt")
dynamics = np.zeros((oommf_data.shape[0], 4))
dynamics[:, 0] = oommf_data[:, -1]  # time
dynamics[:, 1:4] = oommf_data[:, -5:-2]

np.savetxt("averages_oommf.txt", dynamics, header="time mx my mz")
