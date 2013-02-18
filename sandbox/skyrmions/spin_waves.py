import dolfin as df
import numpy as np
import pylab as plt
from finmag.energies import Exchange, DMI, Zeeman
from finmag import Simulation as Sim

tstart = 0
tstatic = 1e-9
tpulse = 0.02e-9
tend = 4e-9

xdim = 7e-9
ydim = 7e-9
zdim = 1e-9

xv = 15
yv = 15

Ms = 1.567e5 #A/m
H_pulse = [0.7*Ms, 0.7*Ms, 0.7*Ms]

sw_alpha = 1e-20

mesh = df.BoxMesh(0, 0, 0, xdim, ydim, zdim, xv, yv, 1)

sim = Sim(mesh, Ms)

sim.set_m((1,0,0))

A = 3.57e-13 #J/m
D = 2.78e-3 #J/m**2
H = [0,0,0]

sim.add(Exchange(A))
sim.add(DMI(D))
sim.add(Zeeman(H))

tsim = np.linspace(tstart, tstatic, 51)

#Simulate to ground state
sim.alpha = 1
for t in tsim:
    sim.run_until(t)
    m = sim.llg._m
    df.plot(m)
            
#Excite the system
tsim = np.linspace(tstatic,tstatic+tpulse,51)
sim.add(Zeeman(H_pulse))
for t in tsim:
    sim.run_until(t)
    m = sim.llg._m
    df.plot(m)

#Record spin waves
tsim = np.linspace(tstatic+tpulse,tend,5001)
sim.alpha = sw_alpha
sim.add(Zeeman([0,0,0]))

xs = np.linspace(0,xdim-1e-22,xv+1)
ys = np.linspace(0,ydim-1e-22,yv+1)
z = 0

mx = np.zeros([xv+1,yv+1,len(tsim)])
my = np.zeros([xv+1,yv+1,len(tsim)])
mz = np.zeros([xv+1,yv+1,len(tsim)])
for i in xrange(len(tsim)):
    sim.run_until(tsim[i])
    m = sim.llg._m
    df.plot(m)
    for j in range(len(xs)):
        for k in range(len(ys)):
            mx[j,k,i] = m(xs[j], ys[k], 0)[0]
            my[j,k,i] = m(xs[j], ys[k], 0)[1]
            mz[j,k,i] = m(xs[j], ys[k], 0)[2]

tsim = tsim - (tstatic+tpulse)

#save the file into the output.npz
np.savez('output', tsim=tsim, mx=mx, my=my, mz=mz)
