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
          self.phi = Function(self.V)

     def solve_phia(self,phia,method = "lu"):
          """
          Solve for potential phia in the Magentic region using FEM.
          By providing a Function phia defined on a space smaller than
          V we can solve domain truncation problems as well. 
          """
          V = phia.function_space()
          v = TestFunction(V)
          
          #Buffer data independant of M
          if not hasattr(self,"formA_phia"):
               #Define functions
               u = TrialFunction(V)
               a = dot(grad(u),grad(v))*dx
               self.phia_formA = assemble(a)
               #Define and apply Boundary Conditions
               self.phia_bc = DirichletBC(V,0,"on_boundary")
               self.phia_bc.apply(self.phia_formA)               
               
          #Source term depends on M
          f = (-div(self.M)*v)*dx  #Source term
          F = assemble(f)
          self.phia_bc.apply(F)
          
          #Solve for phia
          A = self.phia_formA
          solve(A,phia.vector(),F,method)
          
class FemBemGCRSolver(GCRDeMagSolver,sb.FemBemDeMagSolver):
     """FemBem solver for Demag Problems using the GCR approach"""
    
     def __init__(self,problem,degree = 1):
          super(FemBemGCRSolver,self).__init__(problem,degree)
          #get the boundary dof - coordinate dictionary
          self.doftionary = self.get_boundary_dof_coordinate_dict()
          self.normtionary = self.get_dof_normal_dict_avg()

     def solve(self):
          """
          Solve for the Demag field using GCR and FemBem
          Potential is returned, demag field stored
          """
          print "Solve for phia"
          #Solve for phia
          self.solve_phia()
          #Solve for phib on the boundary with Bem 
          self.phib = self.solve_phib_boundary(self.phia,self.doftionary)
          #Solve for phib on the inside of the mesh with Fem
          print "Solve laplace on the inside"
          self.phib = self.solve_laplace_inside(self.phib)
          # Add together the two potentials
          print "Compute phi total"
          self.phi = self.calc_phitot(self.phia,self.phib)

          # FIXME: Why is this here? Uncomment if it breaks the build.
          # It broke the build... TODO: Figure out why this is necessary
          #self.Hdemag = self.get_demagfield(self.phi)
          return self.phi
          
     def solve_phia(self,method = "lu"):
          super(FemBemGCRSolver,self).solve_phia(phia = self.phia,method = method)
          
     def solve_phib_boundary(self,phia,doftionary):
          """Solve for phib on the boundary using BEM"""
          print "Assemble q vector"
          q = self.assemble_qvector_exact()
          if self.bem is None:
              print "B is none, build bem"
              self.bem = self.build_BEM_matrix()
              
          print "Dot product between B and q"
          phibdofs = np.dot(self.bem,q)
          bdofs = doftionary.keys()
          print "Vector assignment"
          for i in range(len(bdofs)):
               self.phib.vector()[bdofs[i]] = phibdofs[i]
          return self.phib

     def build_BEM_matrix(self):
          """Build the BEM Matrix associated to the mesh and store it"""
          info_blue("Calculating BEM matrix")
          dimbem = len(self.doftionary)
          bemmatrix = np.zeros([dimbem,dimbem])

          import progressbar as pb
          bar = pb.ProgressBar(maxval=dimbem-1, \
                 widgets=[pb.ETA(), pb.Bar('=', '[', ']'), ' ', pb.Percentage()])

          for index,dof in enumerate(self.doftionary):
               bar.update(index)
               bemmatrix[index] = self.get_bem_row(self.doftionary[dof],self.doftionary.keys())
               #info("BEM Matrix line "+ str(index) + str(self.bemmatrix[index]))
          return bemmatrix

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
          kwargs = {"R"+str(i):R[i] for i in range(dim)}
          E = Expression(w,**kwargs)
          return E

     def assemble_qvector_exact(self):
          """Builds the vector q using point evaluation"""

          q = np.zeros(len(self.normtionary))
          #Get gradphia as a vector function
          gradphia = project(grad(self.phia), VectorFunctionSpace(self.V.mesh(),"DG",0))
          for i,dof in enumerate(self.doftioynary):
               ri = self.doftionary[dof]
               n = self.normtionary[dof]
               
               #Take the dot product of n with M + gradphia(ri) (n dot (M+gradphia(ri))
               rtup = tuple(ri)
               M_array = np.array(self.M(rtup))
               gphia_array = np.array(gradphia(rtup))
               q[i] = np.dot(n,M_array+gphia_array)
          return q
     
##Not used at the moment
##          def assemble_qvector_average(self,phia = None,doftionary = None):
##          """builds the vector q that we multiply the Bem matrix with to get phib, using an average"""
##          ###At the moment it is advisable to use assemble_qvector_exact as it gives a better result###
##          if phia is None:
##               phia = self.phia
##          V = phia.function_space()
##          if doftionary is None:
##               doftionary = self.get_boundary_dof_coordinate_dict(V)
##          mesh = V.mesh()
##          n = FacetNormal(mesh)
##          v = TestFunction(V)
##          
##          one = assemble(v*ds).array()
##          #build q everywhere v needed so a vector is assembled #This method uses an imprecise average
##          q = assemble((- dot(n,self.M) + dot(grad(phia),n))*v*ds).array()
##          #Get rid of the volume of the basis function
##          basefuncvol = assemble(v*ds).array()
##          #This will create a lot of NAN which are removed by the restriction
##          q = np.array([q[i]/basefuncvol[i] for i in range(len(q))])
##          
##          ########################################
##          #TODO Divide out the volume of the facets
##          ########################################
##          
##          #restrict q to the values on the boundary
##          q = self.restrict_to(q,doftionary.keys())
##          return q
     
if __name__ == "__main__":
     from finmag.demag.problems import prob_fembem_testcases as pft
     problem = pft.MagUnitCircle(10)
     solver = FemBemGCRSolver(problem)
     solver.assemble_qvector_exact()

