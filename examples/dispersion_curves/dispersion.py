import os
import subprocess
import numpy as np
import dolfin as df

from finmag import Simulation as Sim
from finmag.energies import Exchange
from finmag.energies import Demag
from finmag.energies import TimeZeeman

from finmag.util.meshes import from_geofile
            

def relax_system(geo_file):
    mesh = from_geofile(geo_file)
    sim = Sim(mesh, 8.6e5, unit_length=1e-9)
    sim.alpha = 0.5
    sim.set_m((1, 0, 0))

    exchange = Exchange(13.0e-12)
    sim.add(exchange)
    
    demag = Demag(solver="FK")
    sim.add(demag)
    
    sim.relax()
    np.save('relaxed', sim.llg._m.vector().array())  
    
    
def excite_system(geo_file):
    mesh = from_geofile(geo_file)
    sim = Sim(mesh, 8.6e5, unit_length=1e-9)
    sim.alpha = 0.01
    init_m=np.load('relaxed.npy')
    sim.set_m(init_m)

    exchange = Exchange(13.0e-12)
    sim.add(exchange)
    
    demag = Demag(solver="FK")
    sim.add(demag)
    
    llg=sim.llg
    
    mult = df.Function(llg.S1)
    mult.assign(df.Expression("(x[0]>x_limit||x[0]<-x_limit) ? 100.0 : 1.0", x_limit=530))
    sim.spatial_alpha(0.01,mult)
    
    GHz=1e9
    omega= 50*2*np.pi*GHz
    
    field_expr="".join(["(x[0]==0) ?",
              "(t==0?H0",
              ":H0*sin(omega*t)/(omega*t))",
              ":(t==0?H0*sin(kc*x[0])/(kc*x[0])",
              ":H0*sin(kc*x[0])/(kc*x[0])*sin(omega*t)/(omega*t))"])
    
    H = df.Expression(("0.0", field_expr,"0.0"), H0=1e5,kc=1.0,omega=omega, t=0.0)
     
    H_app = TimeZeeman(H)
    H_app.setup(llg.S3, llg._m, Ms=8.6e5)
    
    t0=50e-12
    def update_H_ext(t):
        H_app.update(t-t0)
         
    llg.effective_field.add(H_app, with_time_update=update_H_ext)
    
    xs=np.linspace(-540+1e-8, 540-1e-8, 541)
    times = np.linspace(0, 2e-9, 2001)
    data=[]
    for t in times:
        sim.run_until(t)
        my=np.array([llg._m(x,0,0)[1] for x in xs])
        data.append(my)
        print 'run time  t=%g'%t
    
    np.save('data', np.array(data))


def compute_dispersion(dx,dt,file_name):
    
    data=np.load('data.npy')

    res=np.fft.fft2(data)
    res=np.fft.fftshift(res)
    res=np.abs(res)
    res=np.power(res,2)
    res=np.log10(res)
    m,n=res.shape
    print m,n
    
    freq=np.fft.fftfreq(m,d=dt)
    kx=np.fft.fftfreq(n,d=dx/(2.0*np.pi))
    freq=np.fft.fftshift(freq)
    kx=np.fft.fftshift(kx)

    f=open(file_name,"w")
    f.write('# kx (nm^-1)        frequency (GHz)        FFT_Power (arb. unit)\n')
    
    for j in range(n):
        for i in range(m):
            f.write("%15g      %15g      %15g\n" % (kx[n-j-1], freq[i], res[i][j]))
        f.write('\n')
    f.close()




if __name__ == '__main__':
    if not os.path.exists('relaxed.npy'):
        relax_system('width-modulated_bar.geo')
    
    if not os.path.exists('data.npy'):
        excite_system('width-modulated_bar.geo')
    
    if not os.path.exists('dispersion.dat'):
        compute_dispersion(2,1e-3,"dispersion.dat")
    
    if not os.path.exists('dispersion.png'):
        cmd=('gnuplot','plot.gnu')
        subprocess.check_call(cmd)