import dolfin as df
import numpy as np
import json
from collections import OrderedDict


"""
Problems:
1. We have to know the mesh in advance.
2. Merge loading and saving into single class (r, w flags).
3. Appending data.
4. Looking into the file (we have to use python module).
"""

class SavingData(object):
    def __init__(self, h5filename, npzfilename, jsonfilename, functionspace):
        self.functionspace = functionspace
        self.h5filename = h5filename
        self.npzfilename = npzfilename
        self.h5file = df.HDF5File(df.mpi_comm_world(), self.h5filename, 'w')
        self.jsonfilename = jsonfilename
        
        self.field_index = 0
        self.t_array = []

        self.fieldsDict = {} # dictionary of all field types e.g. 'm', 'H_eff'

        # create json file
        with open(self.jsonfilename, 'w') as jsonfile:
            json.dump(self.fieldsDict, jsonfile, sort_keys=False)
        jsonfile.close()

    def save_field(self, f, field_name, t):
        name = field_name + str(self.field_index)
        self.h5file.write(f, name)
        
        self.t_array.append(t)
        np.savez(self.npzfilename, t_array=self.t_array)

        # todo: method of doing this without having to open
        # and close json file all time?
        if not self.fieldsDict.has_key(field_name):
            self.fieldsDict[field_name] = OrderedDict()

        self.fieldsDict[field_name][name] = t

        with open(self.jsonfilename, 'w') as jsonfile:
            json.dump(self.fieldsDict, jsonfile, sort_keys=False)
        jsonfile.close()

        self.field_index += 1

    def save_mesh(self, name='mesh'):
        self.h5file.write(self.functionspace.mesh(), name)

    def close(self):
        self.h5file.close(df.mpi_comm_world())


class LoadingData(object):
    def __init__(self, h5filename, npzfilename, jsonfilename, functionspace):
        self.functionspace = functionspace
        self.h5filename = h5filename
        self.npzfilename = npzfilename
        self.h5file = df.HDF5File(df.mpi_comm_world(), self.h5filename, 'r')
        self.jsonfilename = jsonfilename

        npzfile = np.load(self.npzfilename)
        self.t_array = npzfile['t_array']

    def load_mesh(self, name='mesh'):
        mesh_loaded = df.Mesh()

        self.h5file.read(mesh_loaded, name, False)

        return mesh_loaded

    def load_field(self, field_name, t):
        index = np.where(self.t_array==t)[0][0]

        name = field_name + str(index)

        f_loaded = df.Function(self.functionspace)

        # todo: removed last parameter, False to make it work.
        # (originally was self.h5file.read(f_loaded, name))
        # Need to check if this is needed...
        self.h5file.read(f_loaded, name)

        return f_loaded

    def load_field_with_json_data(self, field_name, t):

        with open(self.jsonfilename) as jsonfile:
            fieldsDict = json.load(jsonfile, object_pairs_hook=OrderedDict)
        jsonfile.close()

        # todo: this next line is v. bad!
        # The json file format can easily be changed to a more aproppiate one
        # depending on how we actually want to use the data stored in the json file.
        # For example a dictionary of lists rather than the current dictionary of 
        # ordererDictionaries.
        name = str([item[0] for item in fieldsDict[field_name].items() if item[1]==t][0])

        # name = field_name + str(index)

        f_loaded = df.Function(self.functionspace)
        
        # todo: removed last parameter, False to make it work.
        # (originally was self.h5file.read(f_loaded, name))
        # Need to check if this is needed...
        self.h5file.read(f_loaded, name)

        return f_loaded

    def close(self):
        self.h5file.close()
