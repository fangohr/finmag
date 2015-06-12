import dolfin as df
import numpy as np
import json
from collections import OrderedDict


class SavingData(object):
    def __init__(self, h5filename, jsonfilename, functionspace):
        self.functionspace = functionspace
        self.h5filename = h5filename
        self.jsonfilename = jsonfilename

        self.h5file = df.HDF5File(df.mpi_comm_world(), self.h5filename, 'w')
        
        self.field_index = 0
        self.t_array = []

        self.fieldsDict = {} # dictionary of all field types e.g. 'm', 'H_eff'

        # create json file
        with open(self.jsonfilename, 'w') as jsonfile:
            json.dump(self.fieldsDict, jsonfile, sort_keys=False)
        jsonfile.close()

    def save_mesh(self, name='mesh'):
        self.h5file.write(self.functionspace.mesh(), name)

    def save_field(self, f, field_name, t):
        name = field_name + str(self.field_index)
        self.h5file.write(f, name)
        
        self.t_array.append(t)

        if not self.fieldsDict.has_key(field_name):
            self.fieldsDict[field_name] = OrderedDict()

        self.fieldsDict[field_name][name] = t
        self.fieldsDict['fs_family'] = self.functionspace.ufl_element().family()
        self.fieldsDict['dim'] = self.functionspace.ufl_element().value_shape()[0]

        with open(self.jsonfilename, 'w') as jsonfile:
            json.dump(self.fieldsDict, jsonfile, sort_keys=False)
        jsonfile.close()

        self.field_index += 1

    def close(self):
        self.h5file.close()


class LoadingData(object):
    def __init__(self, h5filename, jsonfilename):
        self.h5filename = h5filename
        self.jsonfilename = jsonfilename

        self.h5file = df.HDF5File(df.mpi_comm_world(), self.h5filename, 'r')

        with open(self.jsonfilename) as jsonfile:
            fieldsDict = json.load(jsonfile, object_pairs_hook=OrderedDict)
        jsonfile.close()

        self.mesh = self.load_mesh()

        self.fs_family = fieldsDict['fs_family']
        self.dim = fieldsDict['dim']
        
        self.functionspace = df.VectorFunctionSpace(self.mesh, self.fs_family,
                                                    1, self.dim)

    def load_mesh(self, name='mesh'):
        mesh_loaded = df.Mesh()

        self.h5file.read(mesh_loaded, name, False)

        return mesh_loaded

    def load_field(self, field_name, t):
        with open(self.jsonfilename) as jsonfile:
            fieldsDict = json.load(jsonfile, object_pairs_hook=OrderedDict)
        jsonfile.close()

        name = str([item[0] for item in fieldsDict[field_name].items() if item[1]==t][0])
        
        f_loaded = df.Function(self.functionspace)
        
        # This line causes segmentation fault.
        self.h5file.read(f_loaded, name)
        
        return f_loaded

    def close(self):
        self.h5file.close()
