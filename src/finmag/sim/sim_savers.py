import finmag

from finmag.util.vtk_saver import VTKSaver
from finmag.util.fileio import FieldSaver

#-----------------------------------------------------------------------------------
# npy savers
#-----------------------------------------------------------------------------------

def _save_field_incremental(sim, field_name, filename=None, overwrite=False):
    save_field(sim,field_name, filename, incremental=True, overwrite=overwrite)

def _save_m_incremental(sim, filename=None, overwrite=False):
    save_field(sim,'m', filename, incremental=True, overwrite=overwrite)

def _get_field_saver(sim, field_name, filename=None, overwrite=False, incremental=False):
    if filename is None:
        filename = '{}_{}.npy'.format(sim.sanitized_name, field_name.lower())
    if not filename.endswith('.npy'):
        filename += '.npy'

    s = None
    if sim.field_savers.has_key(filename) and sim.field_savers[filename].incremental == incremental:
        s = sim.field_savers[filename]

    if s is None:
        s = FieldSaver(filename, overwrite=overwrite, incremental=incremental)
        sim.field_savers[filename] = s

    return s

def save_field(sim, field_name, filename=None, incremental=False, overwrite=False, region=None):
    """
    Save the given field data to a .npy file.

    *Arguments*

    field_name : string

        The name of the field to be saved. This should be either 'm'
        or the name of one of the interactions present in the
        simulation (e.g. Demag, Zeeman, Exchange, UniaxialAnisotropy).

    filename : string

        Output filename. If not specified, a default name will be
        generated automatically based on the simulation name and the
        name of the field to be saved. If a file with the same name
        already exists, an exception of type IOError will be raised.

    incremental : bool

    region:

        Some identifier that uniquely identifies a mesh region. This required
        that the method `mark_regions` has been called previously so that the
        simulation knows about the regions and their IDs.

    """
    field_data = sim.get_field_as_dolfin_function(field_name, region=region)
    field_saver = _get_field_saver(sim,field_name, filename, incremental=incremental, overwrite=overwrite)
    field_saver.save(field_data.vector().array())


#-----------------------------------------------------------------------------------
# Scene rendering
#-----------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------
# VTK savers
#-----------------------------------------------------------------------------------
