import json
from collections import OrderedDict

import dolfin as df

mpi_rank =  df.MPI.rank(df.mpi_comm_world())
mpi_size =  df.MPI.size(df.mpi_comm_world())

# convenience variable
is_master = mpi_rank == 0

class Create(object):
    def __init__(self, filename, functionspace):

        self.functionspace = functionspace
        self.h5filename = filename + '.h5'
        self.jsonfilename = filename + '.json'

        print("Debug: ({}/{}) opening file {}".format(mpi_rank, mpi_size, self.h5filename))
        self.h5file = df.HDF5File(df.mpi_comm_world(), self.h5filename, 'w')

        self.field_index = 0
        self.t_array = []

        self.fieldsDict = OrderedDict()

        self.dump_metadata(self.jsonfilename, self.fieldsDict)

    def save_mesh(self, name='mesh'):
        self.h5file.write(self.functionspace.mesh(), name)

    def save_field(self, f, field_name, t):
        name = field_name + str(self.field_index)
        self.h5file.write(f, name)
        self.t_array.append(t)

        if field_name not in self.fieldsDict:
            self.fieldsDict[field_name] = OrderedDict()
            self.fieldsDict[field_name]['data'] = OrderedDict()
            self.fieldsDict[field_name]['metadata'] = OrderedDict()

        self.fieldsDict[field_name]['data'][name] = t
        self.fieldsDict[field_name]['metadata']['family'] = \
            self.functionspace.ufl_element().family()
        self.fieldsDict[field_name]['metadata']['degree'] = \
            self.functionspace.ufl_element().degree()
        if isinstance(self.functionspace, df.VectorFunctionSpace):
            self.fieldsDict[field_name]['metadata']['type'] = 'vector'
            self.fieldsDict[field_name]['metadata']['dim'] = \
                self.functionspace.ufl_element().value_shape()[0]
        elif isinstance(self.functionspace, df.FunctionSpace):
            self.fieldsDict[field_name]['metadata']['type'] = 'scalar'

        # Adding some debug data
        self.fieldsDict[field_name]['metadata']['mpi-size'] = mpi_size
        self.fieldsDict[field_name]['metadata']['mpi-rank'] = mpi_rank

        self.dump_metadata(self.jsonfilename, self.fieldsDict)

        self.field_index += 1

    def dump_metadata(self, filename, data):
        # create json file (only master)
        if is_master:
            print("Debug: ({}/{}) writing json file {}".format(mpi_rank, mpi_size, filename))
            with open(filename, 'w') as jsonfile:
                json.dump(data, jsonfile, indent=True)
            jsonfile.close()

    def close(self):
        self.h5file.close()


class Read(object):
    def __init__(self, filename):
        self.h5filename = filename + '.h5'
        self.jsonfilename = filename + '.json'

        self.h5file = df.HDF5File(df.mpi_comm_world(), self.h5filename, 'r')

    def load_mesh(self, name='mesh'):
        mesh_loaded = df.Mesh()

        self.h5file.read(mesh_loaded, name, False)

        return mesh_loaded

    def get_fields(self):
        with open(self.jsonfilename) as jsonfile:
            fieldsDict = json.load(jsonfile, object_pairs_hook=OrderedDict)
        jsonfile.close()
        fields_list = []
        for item in fieldsDict.items():
            fields_list.append(item[0])
        return fields_list

    def get_times(self, field_name):
        with open(self.jsonfilename) as jsonfile:
            fieldsDict = json.load(jsonfile, object_pairs_hook=OrderedDict)
        jsonfile.close()
        t_list = []
        for item in fieldsDict[field_name]['data'].items():
            t_list.append(item[1])
        return t_list

    def load_field(self, field_name, t):
        with open(self.jsonfilename) as jsonfile:
            fieldsDict = json.load(jsonfile, object_pairs_hook=OrderedDict)
        jsonfile.close()

        self.mesh = self.load_mesh()

        self.fs_type = fieldsDict[field_name]['metadata']['type']
        self.family = fieldsDict[field_name]['metadata']['family']
        self.degree = fieldsDict[field_name]['metadata']['degree']

        if self.fs_type == 'vector':
            self.dim = fieldsDict[field_name]['metadata']['dim']
            self.functionspace = df.VectorFunctionSpace(self.mesh, self.family,
                                                        self.degree, self.dim)
        elif self.fs_type == 'scalar':
            self.functionspace = df.FunctionSpace(self.mesh, str(self.family),
                                                  self.degree)

        name = str([item[0] for item in fieldsDict[field_name]['data'].items()
                    if abs(item[1] - t) < 1e-10][0])

        f_loaded = df.Function(self.functionspace)

        self.h5file.read(f_loaded, name)

        return f_loaded

    def close(self):
        self.h5file.close()
