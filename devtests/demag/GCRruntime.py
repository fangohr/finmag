#A method by method breakdown of the runtimes for the GCR solver.

import dolfin as df
import cProfile as cP
import finmag.util.timings as ti
import numpy as np
from finmag.demag.solver_gcr import FemBemGCRSolver
from finmag.demag.problems.prob_fembem_testcases import MagSphere20

## This returns too much information
##problem = MagSphere20()
##solver = FemBemGCRSolver(problem)
##cP.run('solver.solve()')

class GCRtimings(FemBemGCRSolver):
    """Test the ti.timings of the GCR solver"""
    ###Methods of FemBemGCRSolver###
    
    def __init__(self,problem,degree = 1):
        ti.timings.start("GCRSolver-init")
        r = super(GCRtimings,self).__init__(problem,degree = degree)
        ti.timings.stop("GCRSolver-init")
        return r
        
    def solve_phia(self,method = "lu"):
        ti.timings.start("solve-phia")
        r = super(GCRtimings,self).solve_phia(method = method)
        ti.timings.stop("solve-phia")
        return r
    
    def build_BEM_matrix(self):
        ti.timings.start("build_BEM_matrix")
        r = super(GCRtimings,self).build_BEM_matrix()
        ti.timings.stop("build_BEM_matrix")
        return r
    
    def assemble_qvector_exact(self):
        ti.timings.start("assemble_qvector_exact")
        
        qtime.start("q = np.zeros(len(self.normtionary))")
        q = np.zeros(len(self.normtionary))
        qtime.stop("q = np.zeros(len(self.normtionary))")
        
        #Get gradphia as a vector function
        qtime.start("gradphia = df.project(df.grad(self.phia), VectorFunctionSpace(self.V.mesh(),'DG',0))")
        gradphia = df.project(df.grad(self.phia), df.VectorFunctionSpace(self.V.mesh(),"DG",0))
        qtime.stop("gradphia = df.project(df.grad(self.phia), VectorFunctionSpace(self.V.mesh(),'DG',0))")
        for i,dof in enumerate(self.doftionary):
            qtime.start("ri = self.doftionary[dof]")
            ri = self.doftionary[dof]
            qtime.stop("ri = self.doftionary[dof]")
            qtime.start("n = self.normtionary[dof]")
            n = self.normtionary[dof]
            qtime.stop("n = self.normtionary[dof]")

            #Take the dot product of n with M + gradphia
            qtime.start("q[i] = sum([n[k]*(self.M[k](tuple(ri)) + gradphia[k](tuple(ri))) for k in range(len(n))])")
            rtup = tuple(ri)
            M_array = np.array(self.M(rtup))
            gphia_array = np.array(gradphia(rtup))
            q[i] = np.dot(n,M_array+gphia_array)
            
            ##Original line for comparison
            ##q[i] = sum([n[k]*(self.M[k](tuple(ri)) + gradphia[k](tuple(ri))) for k in range(len(n))])
            qtime.stop("q[i] = sum([n[k]*(self.M[k](tuple(ri)) + gradphia[k](tuple(ri))) for k in range(len(n))])")
        ti.timings.stop("assemble_qvector_exact")
        return q
    
    ###Methods of FemBemDeMagSolver###
    def calc_phitot(self,func1,func2):
        ti.timings.start("calc_phitot")
        r = super(GCRtimings,self).calc_phitot(func1,func2)
        ti.timings.stop("calc_phitot")
        return r
    
    def solve_laplace_inside(self,function):
        ti.timings.start("solve_laplace_inside")
        r = super(GCRtimings,self).solve_laplace_inside(function)
        ti.timings.stop("solve_laplace_inside")
        return r

class GCRtimingsFull(GCRtimings):
    """
    Timings for high in the call hierarchy  methods are called here as well
    The total time no longer equals the wall time
    """
    def solve_phib_boundary(self,phia,doftionary):
        ti.timings.start("solve_phib_boundary")
        r = super(GCRtimings,self).solve_phib_boundary(phia,doftionary)
        ti.timings.stop("solve_phib_boundary")
        return r
    
    def get_bem_row(self,R,bdofs):
        ti.timings.start("get_bem_row")
        r = super(GCRtimings,self).get_bem_row(R,bdofs)
        ti.timings.stop("get_bem_row")
        return r

    def get_dof_normal_dict_avg(self,V = None):
        ti.timings.start("get_dof_normal_dict_avg")
        r = super(GCRtimings,self).get_dof_normal_dict_avg(V = V)
        ti.timings.stop("get_dof_normal_dict_avg")
        return r
    def bemkernel(self,R):
        ti.timings.start("bemkernel")
        r = super(GCRtimings,self).bemkernel(R)
        ti.timings.stop("bemkernel")
        return r
    
    def get_boundary_dofs(self,V):
        ti.timings.start("get_boundary_dofs")
        r = super(GCRtimings,self).get_boundary_dofs(V)
        ti.timings.stop("get_boundary_dofs")
        return r
    
    def get_boundary_dof_coordinate_dict(self,V = None):
        ti.timings.start("get_boundary_dof_coordinate_dict")
        r = super(GCRtimings,self).get_boundary_dof_coordinate_dict(V = V)
        ti.timings.stop("get_boundary_dof_coordinate_dict")
        return r
    
    def get_dof_normal_dict(self,V = None):
        ti.timings.start("get_dof_normal_dict")
        r = super(GCRtimings,self).get_dof_normal_dict(V = V)
        ti.timings.stop("get_dof_normal_dict")
        return r
    
    def restrict_to(self,bigvector,dofs):
        ti.timings.start("restrict_to")
        r = super(GCRtimings,self).restrict_to(bigvector,dofs)
        ti.timings.stop("restrict_to")
        return r
    

#Create an extra timing object for the method assemble_qvector_exact
qtime = ti.Timings()
    
if __name__ == "__main__":
    problem = MagSphere20()
    solver = GCRtimings(problem)
    solver.solve()
    #Print a review of the 15 most time consuming items
    print ti.timings.report_str(15)

