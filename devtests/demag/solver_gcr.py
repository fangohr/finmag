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
          Solve for potential phia in the magentic region using FEM.
          By providing a Function phia defined on a space smaller than
          V we can solve domain truncation problems as well. 
          """
          V = phia.function_space()
          #Define functions
          u = TrialFunction(V)
          v = TestFunction(V)

          #Define forms
          a = dot(grad(u),grad(v))*dx
          f = (-div(M)*v)*dx  #Source term

          #Define Boundary Conditions
          bc = DirichelBC(V,0,"on_boundary")

          #solve for phia
          A,F = assemble(a,f)

          solve(A,phia.vector(),F,method)

class GCRFemBemDeMagSolver(GCRDeMagSolver,sb.FemBemDeMagSolver):
     """FemBem solver for Demag Problems using the GCR approach"""
    
     def __init__(self,problem,degree = 1):
          super(GCRFemBemDeMagSolver,self).__init__(problem,degree)

     def solve_phia(self,method = "lu"):
          super(GCRFemBemDeMagSolver,self).solve_phia(phia = self.phia,method = method)
          
     def solve_phib_boundary(self):
          """Solve for phib on the boundary using BEM (row by row)"""
          #Get the boundary dofs and their coordinates
          dofcoordinates = self.get_boundary_dof_coordinate_dict()
          for index,dof in enumerate(dofcoordinates):
               row = self.get_bem_row(i,x,bdofs)
               
     def build_BEM_matrix(self):
          """Build the BEM Matrix associated to the mesh and store it"""
          dofcoordinates = self.get_boundary_dof_coordinate_dict()
          dimbem = len(dofcoordinates)
          self.bemmatrix = np.zeros([dimbem,dimbem]);
          for index,dof in enumerate(dofcoordinates):
               self.bemmatrix[index] = self.get_bem_row(i,dofcoordinates[dof],dofcoordinates.keys())
               print self.bemmatrix[index]

     def get_bem_row(self,index,R,bdofs):
          """Gets the row of the BEMmatrix associated with the point R"""
          
          w = Expression("1.0/sqrt((%g - x[0])*(%g - x[0]) + (%g - x[1])*(%g - x[1])+(%g - x[2])*(%g - x[2]))"%(R[0],R[0],R[1],R[1],R[2],R[2]))
          psi = TestFunction(self.V)
          L = 1.0/(4*math.pi)*psi*w*ds
          #bigrow contains many 0's for nonboundary dofs
          bigrow = assemble(L)
          #row contains just boundary dofs
          row = np.zeros(len(bdofs))
          for i,key in enumerate(bdofs):
               row[i] = bigrow[key]
          return row
        
if __name__ == "__main__":
     import prob_fembem_testcases as pft
     problem = pft.MagUnitSphere()
     solver = GCRFemBemDeMagSolver(problem)
     solver.build_BEM_matrix()
