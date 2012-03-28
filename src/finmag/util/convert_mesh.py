"""Module for converting meshes into dolfin meshes"""

__author__ = "Anders Johansen"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"
import os, sys, commands
import finmag.mesh.marker as mark

def convert_mesh(inputfile, outputfile=None):
    """
    Convert a .geo file to a .xml.gz file compatible with Dolfin.
    The resulting file is placed in ~/finmag/mesh
    
    *Arguments*
        inputfile (str)
            Name of a .geo file which is compatible with Netgen
        outputfile (str) [optional]
            Name of generated .xml.gz file which is compatible with Dolfin.
            If no name is given, the generated mesh file will have the same
            name as the original .geo file.

    *Return*
        ouputfile
            Complete filename of generated mesh. See Example.

    *Example*
        This example shows the simple case where one converts the file 
        "myfile.geo" to a dolfin compatible file "myfile.xml.gz".
    
        .. code-block:: python

            from finmag.util.convert_mesh import convert_mesh
            convert_mesh("myfile.geo")

        Another example shows that this function could be given directly 
        as input to Dolfin.Mesh. In this case, the resulting mesh
        is stored in the same directory under the name "mymesh.xml.gz".

        .. code-block:: python

            mesh = Mesh(convert_mesh("myfile.geo", "mymesh.xml.gz"))
            plot(mesh, interactive=True)

    .. Note::

        If the outputfile.xml.gz already exists, this is returned, 
        even though it may not be the correct mesh corresponding
        to the .geo file. 

    """
    
    name, type_ = os.path.splitext(inputfile)
    if type_ != '.geo':
        print 'Only .geo files are supported as input.'
        sys.exit(1)

    if outputfile is not None:
        if '.xml.gz' in outputfile:
            outputfile = outputfile.rstrip('.xml.gz')
    else:
        outputfile = name

    outputfilename = "".join([outputfile,".xml.gz"])
    
    if os.path.isfile(outputfilename):
        print "The mesh %s already exists, and is automatically returned." % outputfilename
        return outputfilename
    print 'Using netgen to convert %s.geo to DIFFPACK format...' % name
    netgen_cmd = 'netgen -geofile=%s -meshfiletype="DIFFPACK Format" -meshfile=%s.grid -batchmode' % (inputfile, name)
    status, output = commands.getstatusoutput(netgen_cmd)
    if status not in (0, 34304): # Trouble on my machine, should just be zero.
        print output
        print "netgen failed with exit code", status
        sys.exit(2)
    print 'Done!'

    # Convert to xml using dolfin-convert
    print 'Using dolfin-convert to convert the DIFFPACK file to Dolfin xml...'
    dolfin_conv_cmd = 'dolfin-convert %s.grid %s.xml' % (name, outputfile)
    status, output = commands.getstatusoutput(dolfin_conv_cmd)
    if status != 0: 
        print output
        print "dolfin-convert failed with exit code", status
        sys.exit(3)
    print 'Done!'

    # Compress xml file using gzip
    print 'Compressing mesh...'
    compr_cmd = 'gzip -f %s.xml' % outputfile
    status, output = commands.getstatusoutput(compr_cmd)
    if status != 0:
        print output
        print "gzip failed with exit code", status
        sys.exit(4)
    print 'Done!'

    # Remove redundant files
    print 'Cleaning up...'
    files = ["%s.xml.bak" % outputfile,
             "%s_mat.xml" % outputfile,
             "%s_bi.xml" % outputfile,
             "%s.grid" % name]
    for f in files:
        if os.path.isfile(f):
            os.remove(f)
    print 'Done!'

    # Test final mesh
    print 'Testing...'
    from dolfin import Mesh
    Mesh(outputfilename)

    print 'Success! Mesh is written to %s.' % outputfilename
    return outputfilename

#This class can easily be generalized to suit other types of meshes
class GeoMeshSphereProblem(object):
    """Base class for problems using generated spherical geomeshes"""
    def __init__(self,maxh):
        self.maxh = maxh
        self.generate_spheremesh(maxh)
        
    def generate_spheremesh(self,maxh):
        """
        Creates.geo meshes in the shapes of unit spheres with
        varying mesh fineness.
        """
        #Check if the .gz file already exists
        namegz =  "sphere" + str(maxh) + ".xml.gz"
        #Output the file to the folder finmag.mesh
        meshpath =  os.path.dirname(mark.__file__)
        pathgz = "".join([meshpath,"/",namegz])
        if not os.path.isfile(pathgz):
            pathgeo = "".join([meshpath,"/","sphere", str(maxh),".geo"])
            #Create a geofile
            f = open(pathgeo,"w")
            content = "algebraic3d \n \n \
                       solid main = sphere (0, 0, 0; 10)-maxh="+str(maxh)+" ; \n \n \
                       tlo main;"
            f.write(content)                   
            f.close()

            #call the mesh generation function
            convert_mesh(pathgeo)
            #the file should now be in 
            #pathgeo - "geo" + ".xml.gz"#

            #Delete the geofile file
            print "removing geo file"
            os.remove(pathgeo)
