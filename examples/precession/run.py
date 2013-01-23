import os
import numpy as np
import dolfin as df
import matplotlib.pyplot as plt
from finmag import Simulation
from finmag.energies import Demag, Exchange

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE = os.path.join(MODULE_DIR, 'precession.png')

ts = np.linspace(0, 3e-10)

def run_simulation(do_precession):
    Ms = 0.86e6

    mesh = df.BoxMesh(0, 0, 0, 30e-9, 30e-9, 100e-9, 6, 6, 20)
    sim = Simulation(mesh, Ms)
    sim.set_m((1, 0, 1))
    sim.llg.do_precession = do_precession
    sim.add(Demag())
    sim.add(Exchange(13.0e-12))

    averages = []
    for t in ts:
        sim.run_until(t)
        averages.append(sim.m_average)
    return np.array(averages)

subfigures = ("without precession", "with precession")
figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
for i, subfigure_name in enumerate(subfigures):
    m = zip(* run_simulation(bool(i)))
    for dim in xrange(3):
        axes[i].plot(ts, m[dim], label="m{}".format(chr(120+dim)))
        axes[i].legend()
    axes[i].set_title(subfigure_name)
    axes[i].set_xlabel("time (s)")
    axes[i].set_ylabel("unit magnetisation")
    axes[i].set_ylim([-0.1, 1.0])
figure.savefig(IMAGE)
