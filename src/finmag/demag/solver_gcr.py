"""Solvers for the demagnetization field using the Garcia-Cervera-Roma approach""" 

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
import finmag.demag.solver_base as sb
import math
import numpy as np

#Set allow extrapolation to true#
parameters["allow_extrapolation"] = True
     
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
          
class FemBemGCRSolver(GCRDeMagSolver,sb.FemBemDeMagSolver):
     """FemBem solver for Demag Problems using the GCR approach"""
    
     def __init__(self,problem,degree = 1):
          super(FemBemGCRSolver,self).__init__(problem,degree)
          #get the boundary dof - coordinate dictionary
          self.doftionary = self.get_boundary_dof_coordinate_dict()

     def solve(self):
          """
          Solve for the Demag field using GCR and FemBem
          Potential is returned, demag field stored
          """
          #Solve for phia
          self.solve_phia()
          #Solve for phia 
          self.phib = self.solve_phib_boundary(self.phia,self.doftionary)
          self.phib = self.solve_laplace_inside(self.phib)
          self.phitot = self.calc_phitot(self.phia,self.phib)
          self.Hdemag = self.get_demagfield(self.phitot)
          return self.phitot
          
     def solve_phia(self,method = "lu"):
          super(GCRFemBemDeMagSolver,self).solve_phia(phia = self.phia,method = method)
          
     def solve_phib_boundary(self,phia,doftionary):
          """Solve for phib on the boundary using BEM"""
          q = self.assemble_qvector_exact(phia,doftionary)
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

          import progressbar as pb
          bar = pb.ProgressBar(maxval=dimbem-1, \
                 widgets=[pb.ETA(), pb.Bar('=', '[', ']'), ' ', pb.Percentage()])

          for index,dof in enumerate(doftionary):
               bar.update(index)
               self.bemmatrix[index] = self.get_bem_row(doftionary[dof],doftionary.keys())
               #info("BEM Matrix line "+ str(index) + str(self.bemmatrix[index]))
          return self.bemmatrix

     def get_bem_row(self,R,bdofs):
          """Gets the row of the BEMmatrix associated with the point R,used in the form w"""
          w = self.bemkernel(R)
          psi = TestFunction(self.V)
          L = 1.0/(4*math.pi)*psi*w*ds
          #Bigrow contains many 0's for nonboundary dofs
          bigrow = assemble(L,form_compiler_parameters=self.ffc_options)
          #Row contains just boundary dofs
          row = self.restrict_to(bigrow,bdofs)
          return row
     
     def bemkernel(self,R):
          """Get the kernel of the GCR BEM matrix, adapting it to the dimension of the mesh"""
          w = "1.0/sqrt("
          dim = len(R)
          for i in range(dim):
               w += "(R%d - x[%d])*(R%d - x[%d])"%(i,i,i,i)
               if not i == dim-1:
                    w += "+"
          w += ")"
          #FIXME Make this more elegant and applicable to all dimensions
          kwargs = {"R"+str(i):R[i] for i in range(dim)}
          E = Expression(w,**kwargs)
          return E

     def assemble_qvector_average(self,phia = None,doftionary = None):
          """builds the vector q that we multiply the Bem matrix with to get phib, using an average"""
          ###At the moment it is advisable to use assemble_qvector_exact as it gives a better result###
          if phia is None:
               phia = self.phia
          V = phia.function_space()
          if doftionary is None:
               doftionary = self.get_boundary_dof_coordinate_dict(V)
          mesh = V.mesh()
          n = FacetNormal(mesh)
          v = TestFunction(V)
          
          one = assemble(v*ds).array()
          #build q everywhere v needed so a vector is assembled #This method uses an imprecise average
          q = assemble((- dot(n,self.M) + dot(grad(phia),n))*v*ds).array()
          #Get rid of the volume of the basis function
          basefuncvol = assemble(v*ds).array()
          #This will create a lot of NAN which are removed by the restriction
          q = np.array([q[i]/basefuncvol[i] for i in range(len(q))])
          
          ########################################
          #TODO Divide out the volume of the facets
          ########################################
          
          #restrict q to the values on the boundary
          q = self.restrict_to(q,doftionary.keys())
          return q
     
     def assemble_qvector_exact(self,phia = None,doftionary = None):
          """Builds the vector q using point evaluation"""
          if phia is None:
               phia = self.phia
          V = phia.function_space()
          if doftionary is None:
               doftionary = self.get_boundary_dof_coordinate_dict(V)
                   
          normtionary = self.get_dof_normal_dict_avg()
          q = np.zeros(len(normtionary))
          #Get gradphia as a vector function
          gradphia = project(grad(phia), VectorFunctionSpace(V.mesh(),"DG",0))
          for i,dof in enumerate(doftionary):
               ri = doftionary[dof]
               n = normtionary[dof]
               #Take the dot product of n with M + gradphia
               q[i] = sum([n[k]*(self.M[k](tuple(ri)) + gradphia[k](tuple(ri))) for k in range(len(n))])
          return q
                                      
if __name__ == "__main__":
     from finmag.demag.problems import prob_fembem_testcases as pft
     problem = pft.MagUnitCircle(10)
     solver = FemBemGCRSolver(problem)
     solver.assemble_qvector_exact()

