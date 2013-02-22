import os
import subprocess
import numpy as np
import dolfin as df
from numpy import fft
from finmag import Simulation as Sim
from finmag.energies import Exchange, Demag, TimeZeeman
from finmag.util.meshes import from_geofile

meshfile = "width_modulated_bar.geo"
mesh = from_geofile(meshfile)
x0 = -540; x1 = -x0; dx_high_alpha = 10; xn_probe = x1 + 1;
Ms = 8.6e5
A = 13.0e-12
initial_m_file = "mxyz_0.npy"
m_for_fourier_analysis_file = "my_t.npy"
dispersion_data_file = "dispersion.dat"
dispersion_plot_file = "dispersion.png"

def relax_system():
    """
    Relax the system to obtain the initial magnetisation for the subsequent simulation.

    """
    sim = Sim(mesh, Ms, unit_length=1e-9)
    sim.set_m((1, 0, 0))
    sim.alpha = 1
    sim.llg.do_precession = False
    sim.add(Exchange(A))
    sim.add(Demag(solver="FK")) 
    sim.relax()
    np.save(initial_m_file, sim.m)  
    
def excite_system():
    """
    Excite the relaxed system with the sinc pulse and save m_y to a file.

    """
    sim = Sim(mesh, Ms, unit_length=1e-9)
    alpha_mult = df.interpolate(df.Expression(
        "(x[0] < x_left || x[0] > x_right) ? 100.0 : 1.0",
        x_left=x0+dx_high_alpha, x_right=x1-dx_high_alpha), sim.llg.S1)
    sim.spatial_alpha(0.01, alpha_mult)
    sim.set_m(np.load(initial_m_file))
    sim.add(Exchange(A))
    sim.add(Demag(solver="FK"))
    
    GHz = 1e9
    omega = 50 * 2 * np.pi * GHz
    sinc = (
        "H_0"
        " * (t == 0 ? 1 : sin(omega * t)/(omega * t))"
        " * (x[0] == 0 ? 1 : sin(k_c * x[0])/(k_c * x[0]))")
    H = df.Expression(("0.0", sinc, "0.0"), H_0=1e5, k_c=1.0, omega=omega, t=0.0)
    pulse = TimeZeeman(H) 
    t_0 = 50e-12
    def update_pulse(t):
        pulse.update(t - t_0)
    sim.add(pulse, with_time_update=update_pulse) 
   
    xs = np.linspace(x0 + 1e-8, x1 - 1e-8, xn_probe)
    ts = np.linspace(0, 2e-9, 2001)
    my_along_x_axis_over_time = []
    for t in ts:
        sim.run_until(t)
        my = np.array([sim.llg._m(x, 0, 0)[1] for x in xs])
        my_along_x_axis_over_time.append(my)
        print "Simulation t = {:.2}.".format(t)
    np.save(m_for_fourier_analysis_file, np.array(my_along_x_axis_over_time))

def compute_dispersion(dx, dt):
    """
    Compute the dispersion relation, where *dx* is distance between points
    where m_y was probed in nm, and *dt* is the length of time between
    measurements in ns.

    """
    my = np.load(m_for_fourier_analysis_file)
    transformed = np.log10(np.power(np.abs(fft.fftshift(fft.fft2(my))), 2))
    m, n = transformed.shape
    print m,n
    
    freq = fft.fftshift(fft.fftfreq(m, dt))
    kx = fft.fftshift(fft.fftfreq(n, dx/(2.0*np.pi)))

    with open(dispersion_data_file, "w") as f:
        f.write('# kx (nm^-1)        frequency (GHz)        FFT_Power (arb. unit)\n')
        for j in range(n):
            for i in range(m):
                f.write("%15g      %15g      %15g\n" % (kx[n-j-1], freq[i], transformed[i][j]))
            f.write('\n')

if __name__ == '__main__':
    if not os.path.exists(initial_m_file):
        print "Creating initial magnetisation."
        relax_system()
    
    if not os.path.exists(m_for_fourier_analysis_file):
        print "Running simulation with excitation."
        excite_system()
    
    if not os.path.exists(dispersion_data_file):
        "Computing dispersion relation."
        compute_dispersion(2, 1e-3)
    
    if not os.path.exists(dispersion_plot_file):
        print "Calling gnuplot to create dispersion plot."
        cmd = ('gnuplot', 'plot.gnu')
        subprocess.check_call(cmd)
