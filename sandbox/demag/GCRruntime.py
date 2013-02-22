"""
A method by method breakdown of the runtimes for the GCR solver.

"""
from finmag.util.timings import mtimed, default_timer
from finmag.energies.demag.solver_gcr import FemBemGCRSolver
from finmag.tests.demag.problems.prob_fembem_testcases import MagSphere20

class GCRtimings(FemBemGCRSolver):
    """Test the timings of the GCR solver"""
    ###Methods of FemBemGCRSolver###
  
    @mtimed
    def __init__(self,problem,degree = 1):
        r = super(GCRtimings,self).__init__(problem,degree = degree)
        return r
       
    @mtimed
    def solve_phia(self,method = "lu"):
        r = super(GCRtimings,self).solve_phia(method = method)
        return r
   
    @mtimed
    def build_BEM_matrix(self):
        r = super(GCRtimings,self).build_BEM_matrix()
        return r
   
    @mtimed
    def assemble_qvector_exact(self):
        r = super(GCRtimings,self).assemble_qvector_exact()
        return r
    
    ###Methods of FemBemDeMagSolver###
    @mtimed
    def calc_phitot(self,func1,func2):
        r = super(GCRtimings,self).calc_phitot(func1,func2)
        return r
   
    @mtimed
    def solve_laplace_inside(self,function):
        r = super(GCRtimings,self).solve_laplace_inside(function)
        return r

class GCRtimingsFull(GCRtimings):
    """
    Timings for high in the call hierarchy  methods are called here as well
    The total time no longer equals the wall time
    """
    @mtimed
    def solve_phib_boundary(self,phia,doftionary):
        r = super(GCRtimings,self).solve_phib_boundary(phia,doftionary)
        return r
   
    @mtimed
    def get_bem_row(self,R):
        r = super(GCRtimings,self).get_bem_row(R)   
        return r

    @mtimed
    def get_dof_normal_dict_avg(self,normtionary):
        r = super(GCRtimings,self).get_dof_normal_dict_avg(normtionary)
        return r

    @mtimed
    def bemkernel(self,R):
        r = super(GCRtimings,self).bemkernel(R)
        return r
   
    @mtimed
    def get_boundary_dofs(self,V):
        r = super(GCRtimings,self).get_boundary_dofs(V)
        return r
   
    @mtimed
    def build_boundary_data(self):
        r = super(GCRtimings,self).build_boundary_data()
        return r
   
    @mtimed
    def restrict_to(self,bigvector):
        r = super(GCRtimings,self).restrict_to(bigvector)
        return r
    
if __name__ == "__main__":
    problem = MagSphere20()
    solver = GCRtimings(problem)
    solver.solve()
    #Print a review of the 15 most time consuming items
    print default_timer.report(15)

