#GB Some depricated FEMBEM methods for the GCR solver have been placed here.

class ProxyGCR():    
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

   def get_boundary_dofs(self,V):
     """Gets the dofs that live on the boundary of the mesh
            of function space V"""
        dummybc = df.DirichletBC(V,0,"on_boundary")
        return dummybc.get_boundary_values("pointwise")

    def restrict_to(self,bigvector):
        """Restrict a vector to the dofs in dofs (usually boundary)"""
        vector = np.zeros(len(self.doflist_double))
        #Recast bigvector as a double type array when calling restict_to
        self.crestrict_to(bigvector.array().view(vector.dtype.name),vector,self.bdofs)
        return vector

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
#Unit tests
           def test_get_boundary_dof_coordinate_dict(self):
        """Test the method build_boundary_data- doftionary"""
        #Insert the special function space
        self.solver.V = FunctionSpace(self.solver.problem.mesh,"CG",1)
        #Call the methid
        self.solver.build_boundary_data()
        #test the result
        numdofcalc = len(self.solver.doftionary)
        numdofactual = BoundaryMesh(self.solver.V.mesh()).num_vertices()
        assert numdofcalc == numdofactual,"Error in Boundary Dof Dictionary creation, number of DOFS " +str(numdofcalc)+ \
                                          " does not match that of the Boundary Mesh " + str(numdofactual)
    def test_get_dof_normal_dict(self):
        """Test the method get_dof_normal_dict"""
        V = self.easyspace()
        #insert V into the solver
        self.solver.V = V
        self.solver.build_boundary_data()
        facetdic = self.solver.doftonormal
        coord = self.solver.doftionary
        
        #Tests
        assert len(facetdic[0]) == 2,"Error in normal dictionary creation, 1,1 UnitSquareMesh with CG1 has two normals per boundary dof"
        assert facetdic.keys() == coord.keys(),"error in normal dictionary creation, boundary dofs do not agree with those obtained from \
                                            get_boundary_dof_coordinate_dict"

    def easyspace(self):
        mesh = UnitSquareMesh(1,1)
        return FunctionSpace(mesh,"CG",1)
    def test_get_dof_normal_dict_avg(self):
        """
        Test the method get_dof_normal_dict_avg, see if average normals
        have length one
        """
        V = self.easyspace()
        #insert V into the solver
        self.solver.V = V
        self.solver.build_boundary_data()
        normtionary = self.solver.normtionary
        for k in normtionary:
            assert near(sqrt(np.dot(normtionary[k],normtionary[k].conj())),1),"Failure in average normal calulation, length of\
                                                                                     normal not equal to 1"




##Deprecated code.        
## import instant

##class DeMagSolver(object):
##    """Base class for Demag Solvers"""
##    def __init__(self,problem,degree =1):
##        """problem - Object from class derived from FEMBEMDemagProblem"""
##        self.problem = problem
##        self.degree = degree
##        #Create the space for the potential function
##        self.V = df.FunctionSpace(self.problem.mesh,"CG",degree)
##        #Get the dimension of the mesh
##        self.D = problem.mesh.topology().dim()
##        #Create the space for the Demag Field
##        if self.D == 1:
##            self.Hdemagspace = df.FunctionSpace(problem.mesh,"DG",0)
##        else:
##            self.Hdemagspace = df.VectorFunctionSpace(problem.mesh,"DG",0)
##
##        #Convert M into a function
##        #HF: I think this should always be
##        #Mspace = VectorFunctionSpace(self.problem.mesh,"DG",self.degree,3)
##        #GB: ToDo fix the magnetisation of the problems so Mspace is always 3d
##        #Some work on taking a lower dim Unit Normal and making it 3D is needed
##        if self.D == 1:
##            self.Mspace = df.FunctionSpace(self.problem.mesh,"DG",self.degree)
##        else:
##            self.Mspace = df.VectorFunctionSpace(self.problem.mesh,"DG",self.degree)
##        
##        #Define the magnetisation
##        # At the moment this just accepts strings, tuple of strings
##        # or a dolfin.Function
##        # TODO: When the magnetisation no longer are allowed to be 
##        # one-dimensional, remove " or isinstance(self.problem.M, str)"
##        if isinstance(self.problem.M, tuple) or isinstance(self.problem.M, str):
##            self.M = df.interpolate(Expression(self.problem.M),self.Mspace)
##        elif 'dolfin.functions.function.Function' in str(type(self.problem.M)):
##            self.M = self.problem.M
##        else:
##            raise NotImplementedError("%s is not implemented." \
##                    % type(self.problem.M))
##
##        
##    def get_demagfield(self,phi,use_default_function_space = True):
##        """
##        Returns the projection of the negative gradient of
##        phi onto a DG0 space defined on the same mesh
##        Note: Do not trust the viper solver to plot the DeMag field,
##        it can give some wierd results, paraview is recommended instead
##
##        use_default_function_space - If true project into self.Hdemagspace,
##                                     if false project into a Vector DG0 space
##                                     over the mesh of phi.
##        """
##
##
##        Hdemag = -grad(phi)
##        if use_default_function_space == True:
##            Hdemag = df.project(Hdemag,self.Hdemagspace)
##        else:
##            if self.D == 1:
##                Hspace = df.FunctionSpace(phi.function_space().mesh(),"DG",0)
##            else:
##                Hspace = df.VectorFunctionSpace(phi.function_space().mesh(),"DG",0)
##            Hdemag = df.project(Hdemag,Hspace)
##        return Hdemag
##        
##    def save_function(self,function,name):
##        """
##        The function is saved as a file name.pvd under the folder ~/results.
##        It can be viewed with paraviewer or mayavi
##        """
##        file = File("results/"+ name + ".pvd")
##        file << function
