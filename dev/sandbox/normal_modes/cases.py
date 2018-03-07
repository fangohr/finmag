import dolfin as df

from finmag import Simulation
from finmag.energies import Demag, Exchange, Zeeman
from finmag.util.consts import Oersted_to_SI

class StdProblem(object):
    def __init__(self):
        self.box_size = [200., 200., 5.] # nm
        self.box_divisions = [40,40,1] # divisions

    def setup_sim(self, m0):
        Hz = [8e4, 0, 0] # A/m
        A = 1.3e-11 # J/m
        Ms = 800e3 # A/m
        alpha = 1.

        mesh = df.BoxMesh(0, 0, 0, *(self.box_size + self.box_divisions))

        sim = Simulation(mesh, Ms)
        sim.alpha = alpha
        sim.set_m(m0)
        sim.add(Demag())
        sim.add(Exchange(A))
        sim.add(Zeeman(Hz))

        return sim

    def initial_m(self):
        return [0,0,1]

    def name(self):
        return "stdproblem-%dx%dx%d" % tuple(self.box_divisions)

class Grimsditch2004(object):

    def setup_sim(self, m0):
        SX, SY, SZ = 116, 60, 20
        nx, ny, nz = 29, 15, 5
        # Fe, ref: PRB 69, 174428 (2004)
        # A = 2.5e-6 erg/cm^3
        # M_s = 1700 emu/cm^3
        # gamma = 2.93 GHz/kOe
        Ms = 1700e3
        A = 2.5e-6*1e-5
        gamma_wrong = 2.93*1e6/Oersted_to_SI(1.) # wrong by a factor of 6 (?)
        Hzeeman = [10e3*Oersted_to_SI(1.), 0, 0]

        mesh = df.BoxMesh(0, 0, 0, SX, SY, SZ, nx, ny, nz)

        sim = Simulation(mesh, Ms)
        sim.set_m(m0)
        sim.add(Demag())
        sim.add(Exchange(A))
        sim.add(Zeeman(Hzeeman))

        return sim
