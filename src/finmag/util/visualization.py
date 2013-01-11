import logging
logger = logging.getLogger("finmag")
logger.warning("This module will probably crash when imported from within Finmag, "
               "but the code does work on its own. There seems to be some kind of "
               "weird incompability which needs to be fixed (although I have no "
               "idea what's causing it).")
import os
import IPython
from paraview import servermanager

def render_paraview_scene(vtu_file, outfile):
    if not os.path.exists(vtu_file):
        raise IOError("vtu file '{}' does not exist.".format(vtu_file))
    servermanager.Disconnect()
    servermanager.Connect()
    reader = servermanager.sources.XMLUnstructuredGridReader(FileName=vtu_file)
    reader.UpdatePipeline()
    view = servermanager.CreateRenderView()
    repr = servermanager.CreateRepresentation(reader, view)
    repr.Representation = "Surface With Edges"
    view.ResetCamera()
    
    dataInfo = reader.GetDataInformation()
    pointDataInfo = dataInfo.GetPointDataInformation()
    arrayInfo = pointDataInfo.GetArrayInformation("m")
    array_range = arrayInfo.GetComponentRange(-1)
    lut = servermanager.rendering.PVLookupTable()
    #lut.RGBPoints = [array_range[0], 0.0, 0.0, 1.0, array_range[1], 1.0, 0.0, 0.0]
    lut.RGBPoints = [-0.1643010526895523, 0.23, 0.299, 0.754, 0.16364353895187378, 0.706, 0.016, 0.15]
    lut.VectorMode = "Component"
    lut.VectorComponent = 1
    repr.LookupTable = lut
    repr.ColorArrayName = "m"
    repr.ColorAttributeType = "POINT_DATA"
    reader.UpdatePipelineInformation()
    
    # XXX TODO: we put this import here because at toplevel it seems
    # to connect to a servermanager, which causes problems. Would be
    # good to figure out how to avoid using paraview.simple (and also
    # how to avoid that a separate window opens during WriteImage, but
    # that's only a minor annoyance).
    #import paraview.simple as pv
    #pv.WriteImage(outfile, view=view)
    view.WriteImage(outfile, "vtkPNGWriter", 1)
    servermanager.Disconnect()
    image = IPython.core.display.Image(filename=outfile)
    return image
