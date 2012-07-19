import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

from finmag.energies import Zeeman
from finmag.energies import Demag
from finmag.llb.exchange import Exchange
from finmag.llb.anisotropy import LLBAnisotropy
from finmag.llb.material import Material
from finmag.llb.llb import LLB


def average(m):
    m.shape=(3,-1)
    t=np.average(m, axis=1)
    m.shape=(-1)
    return np.sqrt(t[0]**2+t[1]**2+t[2]**2)

def saveplot(ts,me,filename):
    fig=plt.figure()
    plt.plot(ts,me)
    plt.xlabel('Time (ps)')
    plt.ylabel('me')
    fig.savefig(filename)

def save_me(ts,me,me_input):
    f=open('me.txt','w')
    f.write('#Temperature    me    me_input\n')
    for i in range(len(ts)):
        tmp='%g  %e  %e\n'%(ts[i],
                      me[i],me_input[i])
        f.write(tmp)
    f.close()

def SpinTest(mesh,T):
    mat = Material(mesh, name='FePt')
    mat.set_m((1, 1, 1))
    mat.T = T
    
    llb = LLB(mat)
    llb.alpha=0.1
    llb.set_up_solver()
        
    llb.interactions.append(mat)
    
    max_time = 20e-12
    ts = np.linspace(0, max_time, num=11)
    
    me_average = []
    mx=[]
    mz=[]
    for t in ts:
        llb.run_until(t)
        me_average.append(average(llb.m))
        mx.append(llb.m[0])
        mz.append(llb.m[-1])
    
    """
    saveplot(ts,me_average,'tt.png')
    saveplot(ts,mx,'mx.png')
    saveplot(ts,mz,'mz.png')
    """
    
    return me_average[-1],mat.m_e


def SeriesTemperatureTest(mesh):
    Ts1=[i for i in range(0,600,20)]
    Ts2=[i for i in range(600,700,5)]
    Ts3=[i for i in range(700,1000,20)]
    Ts=Ts1+Ts2+Ts3
    me=[]
    me_input=[]
    for t in Ts:
        print 'temperature at %g'%t
        me1,me2=SpinTest(mesh,t)
        me.append(me1)
        me_input.append(me2)
    
    fig=plt.figure()
    p1,=plt.plot(Ts,me,'.')
    p2,=plt.plot(Ts,me_input,'-')
    plt.xlabel('Temperature (K)')
    plt.ylabel('me')
    plt.legend([p1,p2],['me','me-input'])
    fig.savefig('me.png')
    
    save_me(Ts,me,me_input)
    

def StochasticSpinTest(mesh,T):
    mat = Material(mesh, name='FePt')
    mat.set_m((1, 1, 1))
    mat.T = T
    
    llb = LLB(mat)
    llb.alpha=0.1
    llb.set_up_stochastic_solver(dt=5e-14)
        
    llb.interactions.append(mat)
    
    max_time = 50e-12
    ts = np.linspace(0, max_time, num=501)
    
    me_average = []
    mx=[]
    mz=[]
    for t in ts:
        llb.run_stochastic_until(t)
        me_average.append(average(llb.m))
        mx.append(llb.m[0])
        mz.append(llb.m[-1])
        #print llb.m
    
    print me_average
    
    saveplot(ts,me_average,'tt.png')
    saveplot(ts,mx,'mx.png')
    saveplot(ts,mz,'mz.png')
    
    
    return me_average[-1],mat.m_e
     


if __name__ == '__main__':
    x0 = y0 = z0 = 0
    x1 = 5e-9
    y1 = 5e-9
    z1 = 5e-9
    nx = 1
    ny = 1
    nz = 1
    mesh = df.Box(x0, y0, z0, x1, y1, z1, nx, ny, nz)
    
    #mesh =df.Interval(11,0,1)
    print mesh.coordinates()
   
    #print SpinTest(mesh,658)
    #SeriesTemperatureTest(mesh)
    StochasticSpinTest(mesh,400)
    
    


