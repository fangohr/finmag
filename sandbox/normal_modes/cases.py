from finmag import Simulation

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
