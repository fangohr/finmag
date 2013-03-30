"""This module contains methods for the assembly of the q vector in the gcr method
   at some point this functionality should be implemented in c++ for greater speed."""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

import dolfin as df
import numpy as np

class PEQBuilder(object):
    """Methods for exact q vector assembly"""
    
    def get_boundary_dofs(self,V):
        """Gets the dofs that live on the boundary of the mesh
        of function space V"""
        dummybc = df.DirichletBC(V,0,"on_boundary")
        return dummybc.get_boundary_values()


    #The finmag extention to the dolfin mesh class, OrientedBoundaryMesh
    #could be modified to provide this data
    def get_dof_normal_dict_avg(self,normtionary):
        """
        Provides a dictionary with all of the boundary DOF's as keys
        and an average of facet normal components associated to the DOF as values
        V = FunctionSpace
        """
        #Take an average of the normals in normtionary
        avgnormtionary = {k:np.array([ float(sum(i)) for i in zip(*normtionary[k])]) for k in normtionary}
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

        #Initialize dof-to-normal dictionary
        doftonormal = {}
        doftionary = {}
        #Loop over boundary facets
        for facet in df.facets(mesh):
            cells = facet.entities(d)
            #one cell means we are on the boundary
            if len(cells) ==1:
                #######################################
                #Shared data for normal and coordinates
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
                facetdofs = dm.tabulate_facet_dofs(local_fi)
                #Global numbers of facet dofs
                globaldoffacet = [globaldofcell[ld] for ld in facetdofs]
                #add the facet's normal to every dof it contains
                for gdof in globaldoffacet:
                    n = facet.normal()
                    ntup = tuple([n[i] for i in range(d)])
                    #If gdof not in dictionary initialize a list
                    if gdof not in doftonormal:
                        doftonormal[gdof] = []
                    #Prevent redundancy in Normals (for example 3d UnitCubeMesh CG1)
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

    def build_vector_q_pe(self,m,Ms,phia):
        """Builds the vector q using point evaluation, eq. (5)"""
##        q = np.zeros(len(self.normtionary))
        q = np.zeros(self.V.dim())
        
        #Get gradphia as a vector function
        gradphia = df.project(df.grad(phia), df.VectorFunctionSpace(self.V.mesh(),"DG",0))

##        #build a list of mesh boundary verticies 
##        boundaryvertices = []
##        for i,v in enumerate(df.vertices(self.V.mesh())):
##            if i in self.doftionary:
##                boundaryvertices.append(v.point())
                
        for dof in self.doftionary:
            ri = self.doftionary[dof]
            n = self.normtionary[dof]
            rtup =ri

            try: 
                gphia_array = np.array(gradphia(*rtup))
                M_array = np.array(m(*rtup))
                q[dof] = Ms*np.dot(n,-M_array + gphia_array)
            except:
                q[dof] = self.movepoint(rtup,n,m,Ms,gradphia)
        return q
    
    def movepoint(self,rtup,n,m,Ms,gradphia):
        """
        If point evaluation fails in q vector assembly point
        it is assumbed that the point must be outside of the mesh.
        In this case the point is moved until it is in the mesh again.
        
        In 2-d one can set df.parameters["extrapolate"] = True, however this
        did not work in 3-d at the time of coding.
        """
        contract = 1.0 - 1e-15
        expand = 1.0 + 1e-15
        try:
            rtupnew = (rtup[0]*contract,rtup[1],rtup[2])
            gphia_array = np.array(gradphia(*rtupnew))
            M_array = np.array(m(*rtupnew))
        except:
            try:
                rtupnew = (rtup[0]*expand,rtup[1],rtup[2])
                gphia_array = np.array(gradphia(*rtupnew))
                M_array = np.array(m(*rtupnew))
            except:
                try:
                    rtupnew = (rtup[0],rtup[1]*contract,rtup[2])
                    gphia_array = np.array(gradphia(*rtupnew))
                    M_array = np.array(m(*rtupnew))
                except:
                    try:
                        rtupnew = (rtup[0],rtup[1]*expand,rtup[2])
                        gphia_array = np.array(gradphia(*rtupnew))
                        M_array = np.array(m(*rtupnew))
                    except:
                        try:
                            rtupnew = (rtup[0],rtup[1],rtup[2]*contract)
                            gphia_array = np.array(gradphia(*rtupnew))
                            M_array = np.array(m(*rtupnew))
                        except:
                            try:
                                rtupnew = (rtup[0],rtup[1],rtup[2]*expand)
                                gphia_array = np.array(gradphia(*rtupnew))
                                M_array = np.array(m(*rtupnew))
                            except:
                                raise Exception("Failure in gcr q vector assembly, \
                                                point could not be moved inside the mesh \
                                                please use box method or reprogram \
                                                solver_gcr_qvector.movepoint")
        return Ms*np.dot(n,-M_array + gphia_array)
    
