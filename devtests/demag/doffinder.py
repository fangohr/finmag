"""A Utility Module used to locate the Dofs that lie on a common boundary"""

from dolfin import * 
import numpy as np

class inputerror(Exception):
    def __str__(self):
        return "Can only give Lagrange Element dimension for mesh dimensions 0-3"

def numdoflagelem(q,dim):
    """NUM of DOF LaGrange ELEMents:

    q -   order of the polynomial.

    dim - is the dimension of the space.

    Returns the number of degrees for the lagrange element.

    """

#Functions to give the dimension of Lagrange elements
    if dim == 0:
        return 1
    elif dim == 1:
        return q +1
    elif dim == 2:
        return (q +1)*(q+2)/2
    elif dim == 3:
        return (q+1)*(q+2)*(q+3)/6
    else:
        raise inputerror

def bounddofs(fspace,degree, facetfunc,num):
    """BOUNDary DOFS

    fspace - function space
    degree - polynomial degree of the basis functions
    facetfunc - function that marks the boundary
    num - is the number that has been used to mark the boundary

    returns the set of global indices for all degrees of freedoms on the boundary
    that has been marked with 'num'.
    """

    mesh = fspace.mesh()
    d = mesh.topology().dim()
    dm = fspace.dofmap()
    #degree = fspace.element().degree()
    
    #Array to store the facet dofs.
    cell_dofs = np.zeros(numdoflagelem(degree,d),dtype=np.uintc)
    facet_dofs = np.zeros(numdoflagelem(degree,d -1) ,dtype=np.uintc)

    #Initialize bounddofset
    bounddofs= set([])
    for facet in facets(mesh):
        if facetfunc[facet.index()] == num:
            cells = facet.entities(d)
            # Create one cell (since we have CG)
            cell = Cell(mesh, cells[0])
            #Get the local to global map
            dm.tabulate_dofs(cell_dofs,cell)
            #Get the local index of the facet with respect to given cell
            local_facet_index = cell.index(facet)
            #Get *local* dofs
            dm.tabulate_facet_dofs(facet_dofs, local_facet_index)
            #Add the dofs to the global dof set.
            bounddofs = bounddofs.union(set(cell_dofs[facet_dofs]))
    return bounddofs
    
if __name__ == "__main__":
    degree = 1
    mesh = UnitInterval(10)
    mesh.init()
    V =  FunctionSpace(mesh, "CG", degree)
