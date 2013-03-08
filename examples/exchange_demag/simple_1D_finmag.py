import dolfin as df
from finmag.energies import Exchange
import numpy as np
import matplotlib.pylab as plt
import os, commands

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# run nmag
commands.getstatusoutput("nsim %s --clean" % os.path.join(MODULE_DIR, "simple_1D_nmag.py"))
nd = np.load(os.path.join(MODULE_DIR, "nmag_hansconf.npy"))

# run finmag
mesh = df.IntervalMesh(100, 0, 10e-9)
S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
m = df.interpolate(df.Expression(("cos(x[0]*pi/10e-9)", "sin(x[0]*pi/10e-9)", "0")), S3)

exchange = Exchange(1.3e-11)
exchange.setup(S3, m, Ms=1)

fd = exchange.energy_density()

# draw an ASCII table of the findings
table_border    = "+" + "-" * 8 + "+" + "-" * 64 + "+"
table_entries   = "| {:<6} | {:<20} {:<20} {:<20} |"
table_entries_f = "| {:<6} | {:<20.8f} {:<20.8f} {:<20g} |"
print table_border
print table_entries.format(" ", "min", "max", "delta")
print table_border
print table_entries_f.format("finmag", min(fd), max(fd), max(fd)-min(fd))
print table_entries_f.format("nmag", min(nd), max(nd), max(nd)-min(nd))
print table_border

# draw a plot of the two exchange energy densities
xs = mesh.coordinates().flatten()
figure, (upper_axis, lower_axis) = plt.subplots(2, 1, sharex=True)

upper_axis.plot(xs, fd, "b-", label="finmag")
lower_axis.plot(xs, nd, "r-", label="nmag")
upper_axis.legend().draw_frame(False)
lower_axis.legend().draw_frame(False)

EPSILON = 1e-6 # to make sure the same zoom is used on both parts of the figure
upper_axis.set_ylim(np.mean(fd) - EPSILON, np.mean(fd) + EPSILON)
lower_axis.set_ylim(np.mean(nd) - EPSILON, np.mean(nd) + EPSILON)

# make the plot prettier
# from https://github.com/matplotlib/matplotlib/blob/master/examples/pylab_examples/broken_axis.py
upper_axis.spines['bottom'].set_visible(False)
lower_axis.spines['top'].set_visible(False)
upper_axis.xaxis.tick_top()
upper_axis.tick_params(labeltop="off")
lower_axis.xaxis.tick_bottom()

#plt.show()
plt.savefig(os.path.join(MODULE_DIR, "simple1D.png"))
plt.close()
