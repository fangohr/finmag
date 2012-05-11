"""Solvers for the demagnetization field using the Garcia-Cervera-Roma approach"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

import dolfin as df
import finmag.demag.solver_base as sb
from finmag.util.timings import timings
import math
import logging
import numpy as np
import instant
#Set allow extrapolation to true#
df.parameters["allow_extrapolation"] = True
logger = logging.getLogger(name='finmag')

class FemBemGCRSolver(sb.FemBemDeMagSolver):
    """
    This approach is similar to the :py:class:`FKSolver <finmag.demag.solver_fk.FemBemFKSolver>`
    approach, so we will just comment on the differences between the
    approaches. As before, the magnetic scalar potential is diveded into
    two parts, :math:`\\phi = \\phi_a + \\phi_b`, but the definition of these
    are different. :math:`\\phi_a` is the solution of the inhomogeneous
    Dirichlet problem defined as

    .. math::

        \\Delta \\phi_a(\\vec r) = \\nabla \\vec M(\\vec r) \\qquad \\qquad (1)

    inside the domain, and

    .. math::

        \\phi_a(\\vec r) = 0 \\qquad \\qquad (2)

    on the boundary and outside the domain. This is solved in a similar
    manner as before, with the variational forms and boundary condition
    given by (code-block 1)

    .. code-block:: python

        a = dot(grad(u),grad(v))*dx
        f = (-div(self.M)*v)*dx  # Source term

        #Define Boundary Conditions
        bc = DirichletBC(V,0,"on_boundary")

    The second potential, :math:`\phi_b`, is the solution of the Laplace equation

    .. math::

        \Delta \phi_b = 0 \\quad \\qquad (3)

    inside the domain, its normal derivative has a discontinuity of

    .. math::

        \\bigtriangleup \\left(\\frac{\\partial \\phi_b}
        {\\partial n}\\right) = -n \\cdot \\vec M(\\vec r)
        + \\frac{\\partial \\phi_a}{\\partial n}

    on the boundary and it vanishes at infinity, with
    :math:`\\phi_b(\\vec r) \\rightarrow 0` for
    :math:`\\lvert \\vec r \\rvert \\rightarrow \\infty`.
    As for the Fredkin-Koehler approach, the boundary problem can be solved
    with BEM. Unlike the Fredkin-Koehler approach, where the vector equation
    for the second part of the potential is the product between the boundary
    element matrix and the first potential on the boundary, we now have

    .. math::

        \\Phi_b = \\mathbf{B} \\cdot \\vec Q \\qquad \\qquad (4)

    The vector :math:`\\vec Q` contains the potential values :math:`q` at
    the sites of the surface mesh, with :math:`q` defined as the right hand
    side of the boundary equation,

    .. math::

        q(\\vec r) = -n \\cdot \\vec M(\\vec r) +
        \\frac{\\partial \\phi_a}{\\partial n} \\qquad \\qquad (5)

    This vector is assembled in the function assemble_qvector_exact.
    The values of the boundary element matrix :math:`\\mathbf{B}` is given by

    .. math::

        B_{ij} = \\frac{1}{4\\pi}\\int_{\\Omega_j} \\psi_j(\\vec r)\\frac{1}
        {\\lvert \\vec R_i - \\vec r \\rvert} \\mathrm{d}s. \\qquad \\qquad (6)

    Solving the Laplace equation inside the domain and adding the two
    potentials, is also done in the exact same way as for the Fredkin-Koehler
    approach.

    """

    def __init__(self,problem,degree = 1):
        super(FemBemGCRSolver,self).__init__(problem,degree)
        #Define the potentials
        self.phia = df.Function(self.V)
        self.phib = df.Function(self.V)

        #Countdown by bem assembly
        self.countdown = True
        #build the boundary data (normals and coordinates)
        self.build_boundary_data()
        self.build_crestrict_to()

    def solve(self):
        """
        Solve for the Demag field using GCR and FemBem
        Potential is returned, demag field stored
        """
        logger.debug("GCR: Solving for phi_a")
        #Solve for phia
        self.solve_phia()
        #Solve for phib on the boundary with Bem
        self.phib = self.solve_phib_boundary(self.phia,self.doftionary)
        #Solve for phib on the inside of the mesh with Fem, eq. (3)
        logger.info("GCR: Solve for phi_b (laplace on the inside)")
        self.phib = self.solve_laplace_inside(self.phib,solverparams = self.phibsolverparams)
        #Add together the two potentials
        self.phi = self.calc_phitot(self.phia,self.phib)
        return self.phi
    
    def build_crestrict_to(self):
        #Create the c++ function for restrict_to
        c_code_restrict_to = """
        void restrict_to(int bigvec_n, double *bigvec, int resvec_n, double *resvec, int dofs_n, unsigned int *dofs) {
            for ( int i=0; i<resvec_n; i++ )
                { resvec[i] = bigvec[int(dofs[i])]; }
        }
        """

        args = [["bigvec_n", "bigvec"],["resvec_n", "resvec"],["dofs_n","dofs","unsigned int"]]
        self.crestrict_to = instant.inline_with_numpy(c_code_restrict_to, arrays=args)


    def solve_phia(self):
        """
        Solve for potential phia in the Magentic region using FEM.
        By providing a Function phia defined on a space smaller than
        V we can solve domain truncation problems as well.
        """
        V = self.phia.function_space()

        #Buffer data independant of M
        if not hasattr(self,"formA_phia"):
            #Try to use poisson_matrix if available
            if not hasattr(self,"poisson_matrix"):
                self.build_poisson_matrix()
            # The poisson matrix corresponds to code-block 1 - first line
            self.phia_formA = self.poisson_matrix
            #Define and apply Boundary Conditions
            # Eq.(2) and code-block 1 - last line
            self.phia_bc = df.DirichletBC(V,0,"on_boundary")
            self.phia_bc.apply(self.phia_formA)

        #Source term depends on m (code-block 1 - second line)
        f = (-df.div(self.m)*self.v)*df.dx  #Source term
        F = df.assemble(f)
        self.phia_bc.apply(F)

        #Solve for phia
        A = self.phia_formA
        df.solve(A,self.phia.vector(),F)

    def solve_phib_boundary(self,phia,doftionary):
        """Solve for phib on the boundary using BEM"""
        logger.info("GRC: Assemble q vector")
        q = self.assemble_qvector_exact()
        if self.bem is None:
            logger.info("Building BEM matrix")
            self.bem = self.build_BEM_matrix()

        # eq. (4)
        logger.debug("GRC: Dot product between B and q")
        phibdofs = np.dot(self.bem,q)
        bdofs = doftionary.keys()
        logger.info("GCR: Vector assignment")
        for i in range(len(bdofs)):
            self.phib.vector()[bdofs[i]] = phibdofs[i]
            return self.phib

    def build_BEM_matrix(self):
        """Build the BEM Matrix associated to the mesh and store it"""
        info_blue("Calculating BEM matrix")
        dimbem = len(self.doftionary)
        bemmatrix = np.zeros([dimbem,dimbem])

        if self.countdown == True:
            from finmag.util.progress_bar import ProgressBar
            bar = ProgressBar(dimbem-1)

        for index,dof in enumerate(self.doftionary):
            if self.countdown == True:
                bar.update(index)
                bemmatrix[index] = self.get_bem_row(self.doftionary[dof])
                #info("BEM Matrix line "+ str(index) + str(self.bemmatrix[index]))
        return bemmatrix

    def get_bem_row(self,R):
        """Gets the row of the BEMmatrix associated with the point R, used in the form w"""
        w = self.bemkernel(R)
        # eq. (6) for each row.
        L = 1.0/(4*math.pi)*self.v*w*ds
        #Bigrow contains many 0's for nonboundary dofs
        bigrow = assemble(L,form_compiler_parameters=self.ffc_options)
        #Row contains just boundary dofs
        row = self.restrict_to(bigrow)
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
        """Builds the vector q using point evaluation, eq. (5)"""
        q = np.zeros(len(self.normtionary))
        #Get gradphia as a vector function
        gradphia = df.project(df.grad(self.phia), df.VectorFunctionSpace(self.V.mesh(),"DG",0))
        for i,dof in enumerate(self.doftionary):
            ri = self.doftionary[dof]
            n = self.normtionary[dof]

            #Take the dot product of n with M + gradphia(ri) (n dot (M+gradphia(ri))
            rtup = tuple(ri)
            M_array = np.array(self.m(rtup))
            gphia_array = np.array(gradphia(rtup))
            q[i] = np.dot(n,M_array+gphia_array)
        return q

    def get_boundary_dofs(self,V):
        """Gets the dofs that live on the boundary of the mesh
            of function space V"""
        dummyBC = df.DirichletBC(V,0,"on_boundary")
        return dummyBC.get_boundary_values()

    def get_dof_normal_dict_avg(self,normtionary):
        """
        Provides a dictionary with all of the boundary DOF's as keys
        and an average of facet normal components associated to the DOF as values
        V = FunctionSpace
        """
        #Take an average of the normals in normtionary
        avgnormtionary = {k:np.array([ float(sum(i))/float(len(i)) for i in zip(*normtionary[k])]) for k in normtionary}
        #Renormalize the normals
        avgnormtionary = {k: avgnormtionary[k]/df.sqrt(np.dot(avgnormtionary[k],avgnormtionary[k].conj())) for k in avgnormtionary}
        return avgnormtionary
    
    def build_boundary_data(self):
        """
        Builds two boundary data dictionaries
        1.doftionary key- dofnumber, value - coordinates
        2.normtionary key - dofnumber, value - average of all facet normal components associated to a DOF
        """
        mesh = self.V.mesh()
        #Initialize the mesh data
        mesh.init()
        d = mesh.topology().dim()
        dm = self.V.dofmap()
        boundarydofs = self.get_boundary_dofs(self.V)


        #It is very import that this vector has the right length
        #It holds the local dof numbers associated to a facet
        facetdofs = np.zeros(dm.num_facet_dofs(),dtype=np.uintc)

        #Initialize dof-to-normal dictionary
        doftonormal = {}
        doftionary = {}
        #Loop over boundary facets
        for facet in df.facets(mesh):
            cells = facet.entities(d)
            #one cell means we are on the boundary
            if len(cells) ==1:
                #######################################
                #Shared Data for Normal and coordinates
                #######################################

                #create one cell (since we have CG)
                cell = df.Cell(mesh,cells[0])
                #Local to global map
                globaldofcell = dm.cell_dofs(cells[0])

                #######################################
                #Find  Dof Coordinates
                #######################################

                #Create the cell dofs and see if any
                #of the global numbers turn up in BoundaryDofs
                #If so update doftionary with the coordinates
                celldofcord = dm.tabulate_coordinates(cell)

                for locind,dof in enumerate(globaldofcell):
                    if dof in boundarydofs:
                        doftionary[dof] = celldofcord[locind]

                #######################################
                #Find Normals
                #######################################
                local_fi = cell.index(facet)
                dm.tabulate_facet_dofs(facetdofs,local_fi)
                #Global numbers of facet dofs
                globaldoffacet = [globaldofcell[ld] for ld in facetdofs]
                #add the facet's normal to every dof it contains
                for gdof in globaldoffacet:
                    n = facet.normal()
                    ntup = tuple([n[i] for i in range(d)])
                    #If gdof not in dictionary initialize a list
                    if gdof not in doftonormal:
                        doftonormal[gdof] = []
                    #Prevent redundancy in Normals (for example 3d UnitCube CG1)
                    if ntup not in doftonormal[gdof]:
                        doftonormal[gdof].append(ntup)

            elif len(cells) == 2:
                #we are on the inside so continue
                continue
            else:
                assert 1==2,"Expected only two cells per facet and not " + str(len(cells))

        #Build the average normtionary and save data
        self.doftonormal = doftonormal
        self.normtionary = self.get_dof_normal_dict_avg(doftonormal)
        self.doftionary = doftionary
        #numpy array with type double for use by instant (c++)
        self.doflist_double = np.array(doftionary.keys(),dtype = self.normtionary[self.normtionary.keys()[0]].dtype.name)
        self.bdofs = np.array(doftionary.keys())

    def restrict_to(self,bigvector):
        """Restrict a vector to the dofs in dofs (usually boundary)"""
        vector = np.zeros(len(self.doflist_double))
        #Recast bigvector as a double type array when calling restict_to
        self.crestrict_to(bigvector.array().view(vector.dtype.name),vector,self.bdofs)
        return vector
    
if __name__ == "__main__":
     from finmag.demag.problems import prob_fembem_testcases as pft
     problem = pft.MagUnitCircle(10)
     Mspace = df.VectorFunctionSpace(problem.mesh,"CG",1)
     problem.M = df.interpolate(df.Expression(problem.M),Mspace)
     
     solver = FemBemGCRSolver(problem)
     sol = solver.solve()
     plot(sol)
     interactive()
