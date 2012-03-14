"""Solvers for the demagnetization field using the Garcia-Cervera-Roma approach""" 

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
import solver_base as sb
import math
import numpy as np
infrows = 0
class GCRDeMagSolver(sb.DeMagSolver):
     """Class containing methods shared by GCR solvers"""
     def __init__(self,problem,degree = 1):
          super(GCRDeMagSolver,self).__init__(problem,degree)
          #Define the two potentials
          self.phia = Function(self.V)
          self.phib = Function(self.V)
          self.phitot = Function(self.V)

     def solve_phia(self,phia,method = "lu"):
          """
          Solve for potential phia in the Magentic region using FEM.
          By providing a Function phia defined on a space smaller than
          V we can solve domain truncation problems as well. 
          """
          V = phia.function_space()
          #Define functions
          u = TrialFunction(V)
          v = TestFunction(V)

          #Define forms
          a = dot(grad(u),grad(v))*dx
          f = (-div(self.M)*v)*dx  #Source term

          #Define Boundary Conditions
          bc = DirichletBC(V,0,"on_boundary")

          #solve for phia
          A = assemble(a)
          F = assemble(f)
          solve(A,phia.vector(),F,method)
          
class GCRFemBemDeMagSolver(GCRDeMagSolver,sb.FemBemDeMagSolver):
     """FemBem solver for Demag Problems using the GCR approach"""
    
     def __init__(self,problem,degree = 1):
          super(GCRFemBemDeMagSolver,self).__init__(problem,degree)
          #get the boundary dof - coordinate dictionary
          self.doftionary = self.get_boundary_dof_coordinate_dict(self.V)

     def solve(self):
          """Solve for the Demag field using GCR and FemBem"""
          #Solve for phia
          self.solve_phia()
          #Solve for phia 
          self.phib = self.solve_phib_boundary(self.phia,self.doftionary)
          self.phib = self.solve_laplace_inside(self.phib)
          self.phitot = self.calc_phitot(self.phia,self.phib)
          return self.phitot
          
     def solve_phia(self,method = "lu"):
          super(GCRFemBemDeMagSolver,self).solve_phia(phia = self.phia,method = method)
          
     def solve_phib_boundary(self,phia,doftionary):
          """Solve for phib on the boundary using BEM"""
          q = self.assemble_qvector_average(phia,doftionary)
          B = self.build_BEM_matrix(doftionary)
          phibdofs = np.dot(B,q)
          bdofs = doftionary.keys()
          for i in range(len(bdofs)):
               self.phib.vector()[bdofs[i]] = phibdofs[i]
          return self.phib

     def build_BEM_matrix(self,doftionary):
          """Build the BEM Matrix associated to the mesh and store it"""
          info_blue("Calculating BEM matrix")
          dimbem = len(doftionary)
          self.bemmatrix = np.zeros([dimbem,dimbem])
          for index,dof in enumerate(doftionary):
               self.bemmatrix[index] = self.get_bem_row(doftionary[dof],doftionary.keys())
               info("BEM Matrix line "+ str(index) + str(self.bemmatrix[index]))
          return self.bemmatrix

     def get_bem_row(self,bdofs):
          """Gets the row of the BEMmatrix associated with the point R,used in the form w"""
          w = self.bemkernel(R)
          psi = TestFunction(self.V)
          L = 1.0/(4*math.pi)*psi*w*ds
          #Bigrow contains many 0's for nonboundary dofs
          bigrow = assemble(L,form_compiler_parameters=self.ffc_options)
          bigrow = assemble(L)
          #Row contains just boundary dofs
          row = self.restrict_to(bigrow,bdofs)
          return row
     
     def bemkernel(self,R):
          """Get the kernel of the GCR BEM matrix, adapting it to the dimension of the mesh"""
          #TODO: Rewrite the expression so that it is only compiled once or not at all 
          w = "1.0/sqrt("
          dim = len(R)
          for i in range(dim-1):
               w += "(%g - x[%d])*(%g - x[%d]) + "%(R[i],i,R[i],i)
          w += "(%g - x[%d])*(%g - x[%d]))"%(R[dim-1],dim -1,R[dim-1],dim-1)
          return Expression(w)     

     def assemble_qvector_average(self,phia,doftionary):
          """builds the vector q that we multiply the Bem matrix with to get phib, using an average"""

          V = phia.function_space()
          mesh = V.mesh()
          n = FacetNormal(mesh)
          v = TestFunction(V)
          
          one = assemble(v*ds).array()
          #build q everywhere v needed so a vector is assembled #This method uses an imprecise average
          q = assemble((- dot(n,self.M) + dot(grad(phia),n))*v*ds).array()
          #Get rid of the volume of the basis function
          one = assemble(v*ds).array()
          #This will create a lot of NAN which are removed by the restriction
          q = np.array([q[i]/one[i] for i in range(len(q))])
          #restrict q to the values on the boundary
          q = self.restrict_to(q,doftionary.keys())
          return q
     
     def assemble_qvector_exact(self,phia,doftionary):
          """Builds the vector q using point evaluation"""
          n = FacetNormal(self.problem.mesh)
          v = TestFunction(self.V)
          elist = self.unit_vector_functions(self.problem.mesh())
          #Divide out the test function volumes
          one = assemble(v*ds)
          #assemble normal vector components with a test function in order to get a dofvector
          nlist = [assemble(dot(e,n)*v*ds) for e in elist]
          
          
     def unit_vector_functions(self,mesh):
         """Builds Unit Vector functions defined over the whole mesh"""
         ##uvecspace = VectorFunctionSpace(mesh,"DG",0)
         d = mesh.topology().dim()
         #Create a zero vector"        
         zerovec = [0 for i in range(d)]
         #Initialize unit vector list
         elist = [zerovec[:] for i in range(d)]
         #Change an entry to get a unit vector
         for i in range(d):          
             elist[i][i] = 1
         #Generate constants
         elist = [Constant(tuple(elist[i])) for i in range(len(elist))]
         print elist
         return elist

                        
if __name__ == "__main__":
     import prob_fembem_testcases as pft
     problem = pft.MagUnitCircle()
     solver = GCRFemBemDeMagSolver(problem)
     
##     phitot = solver.solve()
####     plot(phitot)
####     interactive()
##     solver.save_function(solver.phitot, "GCR Solver Solution")
