"""Solvers for the demagnetization field using the Garcia-Cervera-Roma approach""" 

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
import solver_base as sb
import math
import numpy as np

class GCRDeMagSolver(sb.DeMagSolver):
     """Class containing methods shared by GCR solvers"""
     def __init__(self,problem,degree = 1):
          super(GCRDeMagSolver,self).__init__(problem,degree)
          #Define the two potentials
          self.phia = Function(self.V)
          self.phib = Function(self.V)

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
          self.doftionary = self.get_boundary_dof_coordinate_dict()

     def solve(self):
          """Solve for the Demag field using GCR and FemBem"""
          self.solve_phia()
          self.solve_phib_boundary(self.phia,self.doftionary)
          
     def solve_phia(self,method = "lu"):
          super(GCRFemBemDeMagSolver,self).solve_phia(phia = self.phia,method = method)
          
     def solve_phib_boundary(self,phia,doftionary):
          """Solve for phib on the boundary using BEM"""
          q = self.assemble_qvector(phia,doftionary.keys())


     def build_BEM_matrix(self,doftionary):
          """Build the BEM Matrix associated to the mesh and store it"""

          dimbem = len(doftionary)
          self.bemmatrix = np.zeros([dimbem,dimbem]);
          for index,dof in enumerate(doftionary):
               self.bemmatrix[index] = self.get_bem_row(doftionary[dof],doftionary.keys())
               print self.bemmatrix[index]

     def get_bem_row(self,R,bdofs):
          """Gets the row of the BEMmatrix associated with the point R"""
          w = self.bemkernel(R)
          psi = TestFunction(self.V)
          L = 1.0/(4*math.pi)*psi*w*ds
          #Bigrow contains many 0's for nonboundary dofs
          bigrow = assemble(L)
          #Row contains just boundary dofs
          row = self.restrict_to(bigrow,bdofs)
          return row
     
     def bemkernel(self,R):
          """Get the kernel of the BEM matrix, adapting it to the dimension of the mesh"""
          w = "1.0/sqrt("
          dim = len(R-1)
          for i in range(dim-1):
               w += "(%g - x[%d])*(%g - x[%d]) + "%(R[i],i,R[i],i)
          w += "(%g - x[%d])*(%g - x[%d]))"%(R[dim-1],dim -1,R[dim-1],dim-1)
          return Expression(w)     

     def assemble_qvector(self,phia,bdofs):
          """builds the vector q that we multiply the bem matrix with to get phib"""
          #FIXME I believe project attempts to build use a facet normal on the inside and runs into trouble
          qspace = FunctionSpace(self.problem.mesh,"DG",1)
          #build q everywhere
          n = FacetNormal(self.problem.mesh)
          q = - dot(n,self.M) + dot(grad(phia),n)
          q = project(q,qspace)
          #restrict q to the values on the boundary
          ##q = self.restrict_to(q.vector().array(),bdofs)
          plot(q)
          interactive()
          return q
          
    
if __name__ == "__main__":
     import prob_fembem_testcases as pft
     problem = pft.MagUnitCircle()
     solver = GCRFemBemDeMagSolver(problem)
     solver.solve()
