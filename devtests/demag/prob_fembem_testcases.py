"""Standard Demagnetisation Testproblems for FEMBEM"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
import numpy as np
import prob_base as pb

class MagUnitSphere(pb.FemBemDeMagProblem):
    """Uniformly magnetized sphere problem for fembem solvers"""
    def __init__(self):
        mesh = UnitSphere(10)
        M = ("1","0","0")
        #Initialize Base Class
        super(MagUnitSphere,self).__init__(mesh,M)
        
    def desc(self):
        return "unit sphere demagnetisation test problem fembem"
