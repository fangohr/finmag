"""A Utility Module that defines error norms for dolfin functions"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
import numpy as np

def L2_error(f1,f2,cell_domains = None,interior_facet_domains = None, dx = dx):
    """L2 error norm for functions f1 and f2, dx = Measure"""
    Eform = inner(f1-f2,f1-f2)*dx
    E = assemble(Eform, cell_domains =  cell_domains, interior_facet_domains =interior_facet_domains)
    return sqrt(E)

def discrete_max_error(f1,f2):
    """Max discrete error norm (using dofs) for two functions f1 and f2"""
    #Note at the moment Vector functions are flattened out
    v = f1.vector().array() - f2.vector().array()
    M = max(v)
    return M

    
