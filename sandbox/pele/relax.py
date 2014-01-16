import os
import dolfin as df
import numpy as np


from finmag.util import helpers
from finmag.energies.effective_field import EffectiveField
from finmag.energies import Exchange, DMI, UniaxialAnisotropy

from finmag import Simulation as Sim


import logging
log = logging.getLogger(name="finmag")

ONE_DEGREE_PER_NS = 17453292.5  # in rad/s

from pele.potentials import BasePotential

def cartesian2spherical(xyz):
    xyz.shape=(3,-1)
    r_xy = np.sqrt(xyz[0,:]**2 + xyz[1,:]**2)
    theta =  np.arctan2(r_xy, xyz[2,:])
    phi = np.arctan2(xyz[1,:], xyz[0,:])
    xyz.shape=(-1,)
    
    theta_phi = np.concatenate((theta, phi))
    
    return theta_phi

def spherical2cartesian(theta_phi):
    theta_phi.shape=(2,-1)
    theta=theta_phi[0]
    phi = theta_phi[1]
    mxyz = np.zeros(3*len(theta))
    mxyz.shape=(3,-1)
    mxyz[0,:] = np.sin(theta)*np.cos(phi)
    mxyz[1,:] = np.sin(theta)*np.sin(phi)
    mxyz[2,:] = np.cos(theta)
    mxyz.shape=(-1,)
    theta_phi.shape=(-1,)
    return mxyz


def cartesian2spherical_field(field_c,theta_phi):
    
    theta_phi.shape=(2,-1)
    theta = theta_phi[0]
    phi = theta_phi[1]
    
    field_s = np.zeros(theta_phi.shape)
    
    field_c.shape = (3,-1)
    field_s.shape = (2,-1)
    
    hx = field_c[0]
    hy = field_c[1]
    hz = field_c[2]
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    
    field_s[0] = (hx*cos_p + hy*sin_p)*cos_t - hz*sin_t
    field_s[1] = (-hx*sin_p + hy*cos_p)*sin_t
    
    field_c.shape=(-1,)
    field_s.shape=(-1,)
    theta_phi.shape=(-1,)
    
    return field_s

def create_simulation():
    
    from finmag.util.meshes import ellipsoid
    #mesh = df.IntervalMesh(10,0,30)
    #mesh = df.RectangleMesh(0,0,10,2,5,1)
    mesh = ellipsoid(30,10,10,maxh=3.0)
    
    sim = Sim(mesh, Ms=8.6e5, unit_length=1e-9)
    
    sim.set_m((1,1,1))
    sim.add(Exchange(1.3e-11))
    sim.add(UniaxialAnisotropy(-1e5, (0, 0, 1), name='Kp'))
    sim.add(UniaxialAnisotropy(1e4, (1, 0, 0), name='Kx'))
    
    return sim

sim = create_simulation()
m_fun = sim.llg._m
effective_field = sim.llg.effective_field

class My1DPot(BasePotential):
    """1d potential"""
    def getEnergy(self, x):
        m=spherical2cartesian(x)
        m_fun.vector().set_local(m)
        effective_field.update()
        return effective_field.total_energy()
        

    def getEnergyGradient(self, x):
        m=spherical2cartesian(x)
        m_fun.vector().set_local(m)
        effective_field.update()
        E = effective_field.total_energy()
        field = effective_field.H_eff
        grad = cartesian2spherical_field(field, x)
        return E, grad


from pele.systems import BaseSystem
class My1DSystem(BaseSystem):
    def get_potential(self):
        return My1DPot()



if __name__ == '__main__':
    
    sys = My1DSystem()
    database = sys.create_database()
    x0 = cartesian2spherical(sim.llg.m)
    bh = sys.get_basinhopping(database=database, coords=x0)
    bh.run(20)
    print "found", len(database.minima()), "minima"
    min0 = database.minima()[0]
    print "lowest minimum found at", spherical2cartesian(min0.coords), "with energy", min0.energy
    
    for min in database.minima():
        sim.set_m(spherical2cartesian(min.coords))
        df.plot(m_fun)
    
    df.interactive()
    
