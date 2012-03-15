"""A Utility Module used to locate the Dofs that lie on a common boundary"""

from dolfin import * 
import numpy as np

def bounddofs(fspace, facetfunc,num):
    """BOUNDary DOFS

    fspace - function space
    facetfunc - function that marks the boundary
    num - is the number that has been used to mark the boundary

    returns the set of global indices for all degrees of freedoms on the boundary
    that has been marked with 'num'.
    """

    mesh = fspace.mesh()
    d = mesh.topology().dim()
    dm = fspace.dofmap()
    
    #Array to store the facet dofs.
    facet_dofs = np.zeros(dm.num_facet_dofs(),dtype=np.uintc)

    #Initialize bounddofset
    bounddofs= set([])
    for facet in facets(mesh):
        if facetfunc[facet.index()] == num:
            cells = facet.entities(d)
            # Create one cell (since we have CG)
            cell = Cell(mesh, cells[0])
            # Create a vector with length exactly = #dofs in cell
            cell_dofs = np.zeros(dm.cell_dimension(cell.index()),dtype=np.uintc)
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
    import prob_trunc_testcases as pttc
    import solver_base as sb
    
    problem =  pttc.MagUnitCircle()
    solver = sb.DeMagSolver(problem,2)
    bounddofs(solver.V,2,problem.coreboundfunc,problem.COREBOUNDNUM)
