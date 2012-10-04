import numpy as np
import dolfin as df
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mayavi.mlab as ml
from math import pi, ceil
from finmag import Simulation
from finmag.energies import Zeeman, UniaxialAnisotropy, ThinFilmDemag
import finmag.util.helpers as h
from proto_anisotropy import ProtoAnisotropy

# todo: what is the fundamental frequency?

mesh = df.Interval(1, 0, 1e-9)
# not possible to have 0D mesh. Tried PolygonalMeshGenerator and MeshEditor.

Ms_Oersted = 17.8e3; Ms = Ms_Oersted / (pi * 4e-3);
H_anis_Oersted = 986; H_anis = H_anis_Oersted / (pi * 4e-3);
H_ext_Oersted = [1e3, 1.5e3, 2e3]
J_mult = np.linspace(0.1, 1.5, 10)
shown_once = False
w0_over_J = []
inv_t0_over_J = []
for H_i in H_ext_Oersted:
    fig_traj, axes_traj = plt.subplots(nrows=int(ceil(1.0*len(J_mult)/2)), ncols=2, subplot_kw={"projection":"3d"}, figsize=(8, 11))
    fig_mx, axes_mx = plt.subplots(nrows=int(ceil(1.0*len(J_mult)/2)), ncols=2, figsize=(8, 11))
    for i, J_i in enumerate(J_mult):
        H_ext = H_i / (pi * 4e-3)

        sim = Simulation(mesh, Ms)
        # "nearly parallel to fixed layer magnetisation", 1 degree bias
        sim.set_m((0.01234, 0.01234, 0.99985))
        sim.alpha = 0.003
        zeeman = Zeeman((0, 0, - H_ext))
        sim.add(zeeman)
        ua = ProtoAnisotropy(H_anis)
        #ua = UniaxialAnisotropy(K1=-H_anis, axis=(0, 0, 1))
        sim.add(ua)
        tfd = ThinFilmDemag("x", -0.65)
        sim.add(tfd)

        def show_vectors():
            m = sim.m[::2]
            Hz = zeeman.compute_field()[::2]
            Ha = ua.compute_field()[::2]
            Hd = tfd.compute_field()[::2]
            Heff = sim.llg.effective_field.compute(0)[::2]
            Heff /= h.norm(Heff)
            print Heff
            mxHeff = np.cross(m, Heff)
            figure = ml.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
            ml.quiver3d([0], [0], [0], [m[0]], [m[1]], [m[2]], color=(1, 0, 0))
            ml.quiver3d([0], [0], [0], [Heff[0]], [Heff[1]], [Heff[2]], color=(0, 0, 1))
            ml.quiver3d([0], [0], [0], [mxHeff[0]], [mxHeff[1]], [mxHeff[2]], color=(1, 0, 1))
            ml.axes(figure=figure)
            ml.show()
        if not shown_once:
            show_vectors()
            shown_once = True

        J0 = -1e12 # A/m^2 or 10^8 A/cm^2
        sim.set_stt(df.Constant(J_i*J0), polarisation=1.0, thickness=4e-9, direction=(0, 0, 1))

        ts = np.linspace(0, 5e-9, 1000)
        traj = []
        crossed_z_axis = False
        for t in ts:
            mx, my, mz = sim.m[::2]
            traj.append([mx, my, mz])
            if mz <= 0 and crossed_z_axis == False:
                inv_t0_over_J.append(1/t)
                crossed_z_axis = True
            sim.run_until(t)
        trajectory = np.array(traj)
        trajectory = trajectory.reshape(trajectory.size, order="F").reshape((3, -1))

        axes_mx[i/2,i%2].plot(1e9*ts, trajectory[0])
        axes_mx[i/2,i%2].set_title("H={} Oe, J/J0={}".format(H_i, J_i))
        axes_mx[i/2,i%2].set_xlabel("t [ns]")
        axes_mx[i/2,i%2].set_ylabel("m_x")
        axes_mx[i/2,i%2].set_ylim([-1, 1])
        
        axes_traj[i/2,i%2].plot(trajectory[0], trajectory[1], trajectory[2])
        axes_traj[i/2,i%2].set_xlabel("x")
        axes_traj[i/2,i%2].set_ylabel("y")
        axes_traj[i/2,i%2].set_zlabel("z")
        axes_traj[i/2,i%2].set_title("H={} Oe, J/J0={}".format(H_i, J_i))
    fig_traj.tight_layout()
    fig_traj.savefig("trajectory_{}kOe.png".format(H_i/1000))
    fig_mx.tight_layout()
    fig_mx.savefig("mx_{}kOe.png".format(H_i/1000))

inv_t0_over_J = np.array(inv_t0_over_J).reshape((len(H_ext_Oersted), -1))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("J/J0")
ax.set_ylabel("1/t0 [1/ns]")
for i, H in enumerate(H_ext_Oersted):
    ax.plot(J_mult, 1e-9*inv_t0_over_J[i], label="H={} Oe".format(H))
plt.legend()
plt.savefig("inv_t0_over_J.png")

