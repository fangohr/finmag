"""Base Classes for Demagnetization solvers"""
__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
import numpy as np

class DeMagSolver(object):
    """Base class for Demag Solvers"""
    def __init__(self,problem,degree =1):
        """problem - Object from class derived from TruncDemagProblem"""
        self.problem = problem
        self.degree = degree
        #Create the space for the potential function
        self.V = FunctionSpace(self.problem.mesh,"CG",degree)
        
        #Convert M into a function
        #HF: I think this should always be
        #Mspace = VectorFunctionSpace(self.problem.mesh,"DG",self.degree,3)
        #GB: ToDo fix the magnetisation of the problems so Mspace is always 3d
        #Some work on taking a lower dim Unit Normal and making it 3D is needed
        if self.problem.mesh.topology().dim() == 1:
            self.Mspace = FunctionSpace(self.problem.mesh,"DG",self.degree)
        else:
            self.Mspace = VectorFunctionSpace(self.problem.mesh,"DG",self.degree)
        #Define the magnetisation
        self.M = interpolate(Expression(self.problem.M),self.Mspace)
        
    def get_demagfield(self,phi):
        """
        Returns the projection of the negative gradient of
        phi onto a DG0 space defined on the same mesh
        Note: Do not trust the viper solver to plot the DeMag field,
        it can give some wierd results, paraview is recommended instead
        """
        if phi.function_space().mesh().topology().dim() == 1:
            Hdemagspace = FunctionSpace(phi.function_space().mesh(),"DG",0)
        else:
            Hdemagspace = VectorFunctionSpace(phi.function_space().mesh(),"DG",0)
        Hdemag = -grad(phi)
        Hdemag = project(Hdemag,Hdemagspace)
        return Hdemag
        
    def save_function(self,function,name):
        """
        The function is saved as a file name.pvd under the folder ~/results.
        It can be viewed with paraviewer or mayavi
        """
        file = File("results/"+ name + ".pvd")
        file << function

class FemBemDeMagSolver(DeMagSolver):
    """Base Class for FEM/BEM Demag Solvers"""
    def __init__(self,problem,degree = 1):
        super(FemBemDeMagSolver,self).__init__(problem,degree)
        #Paramters to use a quadrature rule that avoids the endpoints
        #of a triangle
        self.ffc_options = {"quadrature_rule":"canonical" \
                            ,"fquadrature_degree":4}
        #Change the Function space to CR
        self.V = FunctionSpace(self.problem.mesh,"CR",degree)
        #Total Function that we solve for
        self.phitot = Function(self.V)
        
    def calc_phitot(self,func1,func2):
        """Add two functions to get phitotal"""
        self.phitot.vector()[:] = func1.vector()[:] + func2.vector()[:]
        return self.phitot
            
    def get_boundary_dofs(self,V):
        """Gets the dofs that live on the boundary of the mesh
            of function space V"""
        dummyBC = DirichletBC(V,0,"on_boundary")
        return dummyBC.get_boundary_values()

    def get_boundary_dof_coordinate_dict(self,V = None):
        """
        Provides a dictionary with boundary DOF's
        key = DOFnumber in the functionspace defined over the entire mesh
        value = coordinates of DOF
        The order that the dofs appear in the dic becomes their col/row
        number in the BEM matrix.
        V = Function Space
        """
        if V is None:
            V = self.V
        #objects needed
        mesh = V.mesh()
        d = mesh.topology().dim()
        dm = V.dofmap()      
        boundarydofs = self.get_boundary_dofs(V)
        mesh.init()
        
        #Build a dictionary with all dofs and their coordinates
        #TODO Optimize me!
        doftionary = {}
        for facet in facets(mesh):
            cells = facet.entities(d)
            if len(cells) == 2:
                continue
            elif len(cells) == 1:
                #create one cell (since we have CG)
                cell = Cell(mesh, cells[0])
                #Create the cell dofs and see if any
                #of the global numbers turn up in BoundaryDofs
                #If so update the BoundaryDic with the coordinates
                celldofcord = dm.tabulate_coordinates(cell)
                globaldofs = dm.cell_dofs(cells[0])
                globalcoord = dm.tabulate_coordinates(cell)

                for locind,dof in enumerate(globaldofs):
                    doftionary[dof] = globalcoord[locind]
                #restrict the doftionary to the boundary
                for x in [x for x in doftionary if x not in boundarydofs]:
                    doftionary.pop(x)
            else:
                assert 1==2,"Expected only two cells per facet and not " + str(len(cells))
        return doftionary

    def get_dof_normal_dict(self,V = None):
        """
        Provides a dictionary with all of the boundary DOF's as keys
        and a list of facet normal components associated to the DOF as values
        V = FunctionSpace
        """
        #Take my own function space as default
        if V is None:
            V = self.V
        mesh = V.mesh()
        mesh.init()
        d = mesh.topology().dim()
        dm = V.dofmap()

        #It is very import that this  vector has the right length
        #It holds the local dof numbers associated to a facet 
        facetdofs = np.zeros(dm.num_facet_dofs(),dtype=np.uintc)

        #Initialize dof-normal dictionary
        doftonormal = {}

        #Loop over boundary facets
        for facet in facets(mesh):
            cells = facet.entities(d)
            #one cell means we are on the boundary
            if len(cells) ==1:
                cell = Cell(mesh,cells[0])
                local_fi = cell.index(facet)
                dm.tabulate_facet_dofs(facetdofs,local_fi)
                #Local to global map
                globaldofcell = dm.cell_dofs(cells[0])
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
        return doftonormal

    def get_dof_normal_dict_avg(self,V = None):
        """
        Provides a dictionary with all of the boundary DOF's as keys
        and an average of facet normal components associated to the DOF as values
        V = FunctionSpace
        """
        normtionary = self.get_dof_normal_dict(V = V)
        #Take an average of the normals in normtionary
        avgnormtionary = {k:np.array([ float(sum(i))/float(len(i)) for i in zip(*normtionary[k])]) for k in normtionary}
        #Renormalize the normals
        avgnormtionary = {k: avgnormtionary[k]/sqrt(np.dot(avgnormtionary[k],avgnormtionary[k].conj())) for k in avgnormtionary}
        return avgnormtionary
    
    def restrict_to(self,bigvector,dofs):
        """Restrict a vector to the dofs in dofs (usually boundary)"""
        vector = np.zeros(len(dofs))
        for i,key in enumerate(dofs):
             vector[i] = bigvector[key]
        return vector
    
    def solve_laplace_inside(self,function):
        """Take a functions boundary data as a dirichlet BC and solve
            a laplace equation"""
        V = function.function_space()
        bc = DirichletBC(V,function, "on_boundary")
        u = TrialFunction(V)
        v = TestFunction(V)

        #Laplace forms
        a = inner(grad(u),grad(v))*dx
        A = assemble(a)

        #RHS = 0
        f = Function(V).vector()
        #Apply BC
        bc.apply(A,f)

        solve(A,function.vector(),f)
        return function
        
class TruncDeMagSolver(DeMagSolver):
    """Base Class for truncated Demag Solvers"""
    def __init__(self,problem,degree = 1):
            super(TruncDeMagSolver,self).__init__(problem,degree)
            
    def restrictfunc(self,function,submesh):
        """
        Restricts a function in a P1 space to a submesh
        using the fact that the verticies and DOFS are
        numbered the same way. A function defined on a P1
        space is returned. This will ONLY work for P1 elements
        For other elements the dolfin FunctionSpace.restrict()
        should be used when it is implemented.
        """
        wholemesh = function.function_space().mesh()

        #Since compute_vertex_map only accepts a SubMesh object we need to
        #create a trivial "SubMesh" of the whole mesh 
        dummymeshfunc = MeshFunction("uint",wholemesh,wholemesh.topology().dim())
        dummymeshfunc.set_all(1)

        #This is actually the whole mesh, but compute_vertex_map, only accepts a SubMesh
        wholesubmesh = SubMesh(wholemesh,dummymeshfunc,1)
        #Mapping from the wholesubmesh to the wholemesh
        map_to_mesh = wholesubmesh.data().mesh_function("parent_vertex_indices")

        #This is a dictionary mapping the matching DOFS from a Submesh to a SubMesh
        vm = compute_vertex_map(submesh,wholesubmesh) 
        #Now we want to "restrict" the function to the restricted space
        restrictedspace = FunctionSpace(submesh,"CG",1)
        restrictedfunction = Function(restrictedspace)
        for index,dof in enumerate(restrictedfunction.vector()):
            restrictedfunction.vector()[index] = function.vector()[map_to_mesh[vm[index]]]
        return restrictedfunction
