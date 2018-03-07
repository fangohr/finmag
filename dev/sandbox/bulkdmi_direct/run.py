import os
import logging
import pylab as p
import numpy as np
import dolfin as df
df.parameters.reorder_dofs_serial = False

from dolfin import *


from finmag import Simulation as Sim
from finmag.field import Field
from finmag.energies import Exchange, Demag, DMI
from finmag.util.meshes import from_geofile, mesh_volume


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REL_TOLERANCE = 5e-4
Ms = 8.6e5
unit_length = 1e-9
mesh = from_geofile(os.path.join(MODULE_DIR, "cylinder.geo"))

def init_m(pos):
    x, y, z = pos
    r = np.sqrt(x*x+y*y+z*z)
    return 0, np.sin(r), np.cos(r)


def exact_field(pos):
    x, y, z = pos
    r = np.sqrt(x*x+y*y+z*z)
    factor = 4e-3/(4*np.pi*1e-7*Ms)/unit_length
    return -2*factor*np.array([-(z*np.cos(r)+y*np.sin(r))/r, x*np.sin(r)/r, x*np.cos(r)/r])


class HelperExpression(df.Expression):
    
    def eval(self, value, x):
        value[:] = exact_field(x)[:]

    def value_shape(self):
        return (3,)


def run_finmag():

    sim = Sim(mesh, Ms, unit_length=unit_length)
    sim.alpha = 0.5
    sim.set_m(init_m)

    exchange = Exchange(13.0e-12)
    sim.add(exchange)

    dmi = DMI(4e-3)
    sim.add(dmi)

    dmi_direct = DMI(4e-3, method='direct', name='dmi_direct')
    sim.add(dmi_direct)

    df.plot(sim.m_field.f, title='m')

    fun = df.interpolate(HelperExpression(), sim.S3)
    df.plot(fun, title='exact')
    
    df.plot(Field(sim.S3, dmi.compute_field()).f, title='dmi_petsc')
    df.plot(Field(sim.S3, dmi_direct.compute_field()).f, title='dmi_direct', interactive=True)





    

if __name__ == '__main__':

    run_finmag()

