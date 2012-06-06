__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

import dolfin as df
import numpy as np

class PEQBuilder(object):
    """Methods for exact q vector assembly"""
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
        assert len(facetdic[0]) == 2,"Error in normal dictionary creation, 1,1 UnitSquare with CG1 has two normals per boundary dof"
        assert facetdic.keys() == coord.keys(),"error in normal dictionary creation, boundary dofs do not agree with those obtained from \
                                            get_boundary_dof_coordinate_dict"
        
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
