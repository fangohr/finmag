import time
import logging
import dolfin as df
import numpy as np
from finmag.util.timings import timings
from finmag.energies.fkdemag import FKDemag

log = logging.getLogger(name="finmag")

class Simulation(object):
    def __init__(self, mesh, Ms, unit_length=1):
        timings.reset()
        timings.start("Sim-init")

        log.info("Creating Sim object (rank={}/{}) [{}].".format(
            df.MPI.process_number(), df.MPI.num_processes(), time.asctime()))
        log.debug(mesh)

        self.mesh = mesh
        self.Ms = Ms
        self.unit_length = unit_length
        self.S1 = df.FunctionSpace(mesh, "Lagrange", 1)
        self.S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
        self.Volume = df.assemble(df.Constant(1) * df.dx, mesh=mesh)
        self._m = df.Function(self.S3)
        self.t = 0

        self.interactions = []
        self.add(FKDemag())

        timings.stop("Sim-init")


    def add(self, interaction):
        interaction.setup(self.S3, self._m, self.Ms, self.unit_length)
        self.interactions.append(interaction)

    def compute_effective_field(self):
        H_eff = np.array(self._m.shape)
        
        for interaction in self.interactions:
            H_eff += interaction.compute_field(self.m, self.t)
    
        return H_eff

    

