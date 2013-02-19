import dolfin as df
import numpy as np
import pylab as plt
from finmag.energies import Exchange, Zeeman, DMI
from finmag import Simulation as Sim
from math import pi

def plot_excitation(tsinc, Hsinc):
    """Plot the external excitation signal both in time and frequency domain."""
    #time domain
    plt.subplot(211)
    plt.plot(tsinc, Hsinc)
    #frequency domain
    s_excitation = np.fft.fft(Hsinc)
    f_excitation = np.fft.fftfreq(len(s_excitation), d=tsinc[1]-tsinc[0])
    plt.subplot(212)
    plt.plot(f_excitation, np.absolute(s_excitation))
    plt.xlim([-2e12,2e12])
    plt.show()

#output npz file name
file_name = 'sw_dmi'

tstart = 0
#time to converge into the static state
tstatic = 1e-9 #1ns
#excitation pulse duration
tpulse = 0.08e-9
#end time for recording oscilations
tend = 5e-9
#number of steps for recording oscilations
n_osc_steps = 5001
#number of time steps in excitation signal
n_sinc_steps = 501
toffset = 1e-20

#dimensions of the square thin-film sample
xdim = 7e-9
ydim = 7e-9
zdim = 1e-9

#mesh size in x, y and z directions
xv = 5
yv = 5
zv = 1

#magnetisation saturation
Ms = 1.567e5 #A/m

#simulated frequency maximum
fc = 200e9 #200GHz
#excitation signal amplitude
Hamp = 0.07*Ms
# Hsinc = f(tsinc) for the duration of tpulse
tsinc = np.linspace(-tpulse/2, tpulse/2, n_sinc_steps)
Hsinc = Hamp * np.sinc(2*pi*fc*tsinc)

#Gilbert damping for the spin waves recording
sw_alpha = 1e-20

mesh = df.BoxMesh(0, 0, 0, xdim, ydim, zdim, xv, yv, zv)
sim = Sim(mesh, Ms)
sim.set_m((1,1,1))

#exchange energy constant
A = 3.57e-13 #J/m
#DMI constant
D = 2.78e-3 #J/m**2
#external magnetic field
H = [0,0,0] #A/m

sim.add(Exchange(A)) #exchnage interaction
sim.add(DMI(D)) #DMI interaction
sim.add(Zeeman(H)) #Zeeman interaction

############################################################
#time series for the static state simulation
tsim = np.linspace(tstart, tstatic, 101)
#simulation to the ground state
sim.alpha = 1 #dynamics neglected
for t in tsim:
    sim.run_until(t)
    df.plot(sim.llg._m)
############################################################

############################################################
#excite the system with an external sinc field
tsim = np.linspace(tstatic+toffset, tstatic+tpulse, n_sinc_steps)
i = 0 #index for the extrenal excitation
for t in tsim:
    H = [Hsinc[i], Hsinc[i], Hsinc[i]]
    sim.add(Zeeman(H))
    sim.run_until(t)
    df.plot(sim.llg._m)
    i += 1
############################################################

############################################################
#record spin waves
tsim = np.linspace(tstatic+tpulse+toffset, tend, n_osc_steps)
#decrease the Gilbert damping to previously chosen value
sim.alpha = sw_alpha
#turn off an external field
sim.add(Zeeman([0,0,0]))
#points at which the magnetisation is recorded
xs = np.linspace(0, xdim-1e-20, xv+1)
ys = np.linspace(0, ydim-1e-20, yv+1)
z = 0 #magnetisation read only in one x-y plane at z = 0

#make empty arrays for mx, my and mz recording
#don't remember why tsim-1
mx = np.zeros([xv+1, yv+1, n_osc_steps])
my = np.zeros([xv+1, yv+1, n_osc_steps])
mz = np.zeros([xv+1, yv+1, n_osc_steps])

for i in xrange(len(tsim)):
    #simulate up to next time step
    sim.run_until(tsim[i])
    df.plot(sim.llg._m)
    #record the magnetisation state
    for j in range(len(xs)):
        for k in range(len(ys)):
            mx[j,k,i] = sim.llg._m(xs[j], ys[k], 0)[0]
            my[j,k,i] = sim.llg._m(xs[j], ys[k], 0)[1]
            mz[j,k,i] = sim.llg._m(xs[j], ys[k], 0)[2]
#############################################################


tsim = tsim - (tstatic+tpulse+toffset)

#save the file into the file_name.npz file
np.savez(file_name, tsim=tsim, mx=mx, my=my, mz=mz)
