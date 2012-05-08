import time
import logging
import dolfin as df
import numpy as np
import finmag.sim.helpers as h
from finmag.sim.llg import LLG
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
        self.llg = LLG(mesh, unit_length=unit_length, called_from_sim=True)
        self.S1 = df.FunctionSpace(mesh, "Lagrange", 1)
        self.S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
        self.Volume = df.assemble(df.Constant(1) * df.dx, mesh=mesh)
        self._m = df.Function(self.S3)
        self.t = 0

        self.interactions = []
        self.add(FKDemag())

        timings.stop("Sim-init")
    
    def set_m(self, value, **kwargs):
        """
        Set the magnetisation (scaled automatically).
        
        You can either provide a dolfin.Constant or a dolfin.Expression
        directly, or the ingredients for either, i.e. a tuple of numbers
        or a tuple of strings (with keyword arguments if needed), or provide
        the nodal values directly as a numpy array.

        You can call this method anytime during the simulation. However, when
        providing a numpy array during time integration, the use of
        the attribute m instead of this method is advised for performance
        reasons and because the attribute m doesn't normalise the vector.

        """
        if isinstance(value, tuple):
            if isinstance(value[0], str):
                # a tuple of strings is considered to be the ingredient
                # for a dolfin expression, whereas a tuple of numbers
                # would signify a constant
                val = df.Expression(value, **kwargs)
            else:
                val = df.Constant(value)
            new_m = df.interpolate(val, self.S3)
        elif isinstance(value, (df.Constant, df.Expression)):
            new_m = df.interpolate(value, self.S3)
        elif isinstance(value, (list, np.ndarray)):
            new_m = df.Function(self.S3)
            new_m.vector()[:] = value
        else:
            raise AttributeError
        self._m.vector()[:] = h.fnormalise(new_m.vector().array())

    def add(self, interaction):
        interaction.setup(self.S3, self._m, self.Ms, self.unit_length)
        self.interactions.append(interaction)

    def compute_effective_field(self):
        H_eff = np.zeros(self._m.vector().array().shape)
        for interaction in self.interactions:
            H_eff += interaction.compute_field()
        return H_eff

    def compute_dmdt(self):
        return self.llg.compute_dmdt(
            self._m.vector().array(), self.compute_effective_field()) 
        
