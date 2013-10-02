import os
import dolfin as df
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


from finmag import Simulation as Sim
from finmag.energies import Exchange, UniaxialAnisotropy

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def sech(x):
    return 1/np.cosh(x)

def init_m(pos):
    x=pos[0]

    delta = np.sqrt(13e-12/520e3)*1e9
    sx = -np.tanh((x-50)/delta)
    sy = sech((x-50)/delta)
    return (sx,sy,0)


def field_at(pos):
    
    delta = np.sqrt(13e-12/520e3)*1e9
    x = (pos[0]-50)/delta
    
    fx = -sech(x)**2/delta*1e9
    fy = -np.tanh(x)*sech(x)/delta*1e9
    
    return (fx,fy,0)

def init_J(pos):
    
    return (1e12,0,0)

def test_zhangli():
    
    mesh = df.BoxMesh(0, 0, 0, 100, 1, 1, 50, 1, 1)
    
    sim = Sim(mesh, Ms=8.6e5, unit_length=1e-9)
    sim.set_m(init_m)
    
    sim.add(UniaxialAnisotropy(K1=520e3, axis=[1, 0, 0]))
    sim.add(Exchange(A=13e-12))
    sim.alpha = 0.01
    
    sim.set_zhangli(init_J, 0.5,0.01)
    
    p0=sim.m_average
    
    sim.run_until(1e-11)
    p1=sim.m_average

    assert p1[0] < p0[0]
    assert abs(p0[0])<1e-15
    assert abs(p1[0])>1e-3

def init_J_x(pos):
    return (1,0,0)

def compare_gradient_field1():
    
    mesh = df.BoxMesh(0, 0, 0, 100, 1, 1, 50, 1, 1)
    sim = Sim(mesh, Ms=8.6e5, unit_length=1e-9)
    sim.set_m(init_m)
    sim.set_zhangli(init_J_x, 0.5,0.01)
    
    coords = mesh.coordinates()
    
    field = sim.llg.compute_gradient_field()
    v = df.Function(sim.S3)
    v.vector().set_local(field)
    
    f2 = field.copy()
    f2.shape=(3,-1)
    print f2
    i = 0
    for c in coords:
        print c, field_at(c)
        f2[0][i],f2[1][i],f2[2][i]=field_at(c)
        i+=1
    f2.shape=(-1,)
    v2 = df.Function(sim.S3)
    v2.vector().set_local(f2)
    df.plot(v)
    df.plot(v2)
    
    
    
    
    df.interactive()
    print field

def init_J_xy(pos):
    return (1,2,0)

def init_m2(pos):
    x,y=pos[0],pos[1]

    sx = np.sin(x)*np.cos(y)
    sy = np.sqrt(1-sx**2)
    
    return (sx,sy,0)

def field_at2(pos):
    x,y=pos[0],pos[1]
    jx=1
    jy=2
    
    mx = np.sin(x)*np.cos(y)
    my = np.sqrt(1-mx**2)
    
    mx_x = np.cos(x)*np.cos(y)
    mx_y = -np.sin(x)*np.sin(y)
    
    my_x = -0.5*np.sin(2*x)*np.cos(y)**2/my
    my_y = 0.5*np.sin(x)**2*np.sin(2*y)/my
    
    fx = jx*mx_x+ jy*mx_y
    fy = jx*my_x+ jy*my_y
    
    return (fx,fy,0)

def compare_gradient_field2():
    
    mesh = df.BoxMesh(0, 0, 0, np.pi/2, np.pi/2, 1, 20, 20, 1)
    sim = Sim(mesh, Ms=8.6e5, unit_length=1)
    sim.set_m(init_m2)
    sim.set_zhangli(init_J_xy, 0.5,0.01)
    
    coords = mesh.coordinates()
    
    field = sim.llg.compute_gradient_field()
    v = df.Function(sim.S3)
    v.vector().set_local(field)
    
    f2 = field.copy()
    f2.shape=(3,-1)
    i = 0
    for c in coords:
        tx,ty,tz = field_at2(c)
        f2[0][i],f2[1][i],f2[2][i]=field_at2(c)
        i+=1
    f2.shape=(-1,)
    v2 = df.Function(sim.S3)
    v2.vector().set_local(f2)
    df.plot(v)
    df.plot(v2)
    
    
    print np.abs(field-f2)
    print f2
    
    df.interactive()

if __name__ == "__main__":
    #test_zhangli()
    compare_gradient_field2()


