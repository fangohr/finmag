#A method by method breakdown of the runtimes for the GCR solver.

import dolfin as df
import cProfile as cP
from finmag.util.timings import timings
from finmag.demag.solver_gcr import FemBemGCRSolver
from finmag.demag.problems.prob_fembem_testcases import MagSphere20

## This returns too much information
##problem = MagSphere20()
##solver = FemBemGCRSolver(problem)
##cP.run('solver.solve()')


class GCRTimings(FemBemGCRSolver):
    """Test the timings of the GCR solver"""
    ###Methods of FemBemGCRSolver###
    
    def __init__(self,problem,degree = 1):
        timings.start("Demag-init")
        r = super(GCRTimings,self).__init__(problem,degree = degree)
        timings.stop("Demag-init")
        return r
        
    def solve_phia(self,method = "lu"):
        timings.start("solve-phia")
        r = super(GCRTimings,self).solve_phia(method = method)
        timings.stop("solve-phia")
        return r
          
    def solve_phib_boundary(self,phia,doftionary):
        timings.start("solve_phib_boundary")
        r = super(GCRTimings,self).solve_phib_boundary(phia,doftionary)
        timings.stop("solve_phib_boundary")
        return r

    def build_BEM_matrix(self,doftionary):
        timings.start("build_BEM_matrix")
        r = super(GCRTimings,self).build_BEM_matrix(doftionary)
        timings.stop("build_BEM_matrix")
        return r

    def get_bem_row(self,R,bdofs):
        timings.start("get_bem_row")
        r = super(GCRTimings,self).get_bem_row(R,bdofs)
        timings.stop("get_bem_row")
        return r
    
    def bemkernel(self,R):
        timings.start("bemkernel")
        r = super(GCRTimings,self).bemkernel(R)
        timings.stop("bemkernel")
        return r
    
    def assemble_qvector_exact(self,phia = None,doftionary = None):
        timings.start("assemble_qvector_exact")
        r = super(GCRTimings,self).assemble_qvector_exact(phia = phia,doftionary = doftionary)
        timings.stop("assemble_qvector_exact")
        return r

    ###Methods of FemBemDeMagSolver###
    def calc_phitot(self,func1,func2):
        timings.start("calc_phitot")
        r = super(GCRTimings,self).calc_phitot(func1,func2)
        timings.stop("calc_phitot")
        return r
    
    def get_boundary_dofs(self,V):
        timings.start("get_boundary_dofs")
        r = super(GCRTimings,self).get_boundary_dofs(V)
        timings.stop("get_boundary_dofs")
        return r
    
    def get_boundary_dof_coordinate_dict(self,V = None):
        timings.start("get_boundary_dof_coordinate_dict")
        r = super(GCRTimings,self).get_boundary_dof_coordinate_dict(V = V)
        timings.stop("get_boundary_dof_coordinate_dict")
        return r
    
    def get_dof_normal_dict(self,V = None):
        timings.start("get_dof_normal_dict")
        r = super(GCRTimings,self).get_dof_normal_dict(V = V)
        timings.stop("get_dof_normal_dict")
        return r
    
    def get_dof_normal_dict_avg(self,V = None):
        timings.start("get_dof_normal_dict_avg")
        r = super(GCRTimings,self).get_dof_normal_dict_avg(V = V)
        timings.stop("get_dof_normal_dict_avg")
        return r
    
    def restrict_to(self,bigvector,dofs):
        timings.start("restrict_to")
        r = super(GCRTimings,self).restrict_to(bigvector,dofs)
        timings.stop("restrict_to")
        return r
    
    def solve_laplace_inside(self,function):
        timings.start("solve_laplace_inside")
        r = super(GCRTimings,self).solve_laplace_inside(function)
        timings.stop("solve_laplace_inside")
        return r
    
if __name__ == "__main__":
    problem = MagSphere20()
    solver = GCRTimings(problem)
    solver.solve()
    #Print a review of the 15 most time consuming items
    print timings.report_str(15)

