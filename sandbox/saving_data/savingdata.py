import dolfin as df
import numpy as np


class SavingData(object):
    def __init__(self, h5filename, npzfilename, functionspace):
        self.functionspace = functionspace
        self.h5filename = h5filename
        self.npzfilename = npzfilename
        self.h5file = df.HDF5File(functionspace.mesh().mpi_comm(), h5filename, 'w')
        
        self.field_index = 0
        self.t_array = []

    def save_field(self, f, field_name, t):
        name = field_name + str(self.field_index)
        self.h5file.write(f, name)
        
        self.t_array.append(t)
        np.savez(self.npzfilename, t_array=self.t_array)

        self.field_index += 1

    def save_mesh(self, name='mesh'):
        self.h5file.write(self.functionspace.mesh(), name)

    def close(self):
        self.h5file.close()


class LoadingData(object):
    def __init__(self, h5filename, npzfilename, functionspace):
        self.functionspace = functionspace
        self.h5filename = h5filename
        self.npzfilename = npzfilename
        self.h5file = df.HDF5File(functionspace.mesh().mpi_comm(), h5filename, 'r')
        
        npzfile = np.load(npzfilename)
        self.t_array = npzfile['t_array']

    def load_mesh(self, name='mesh'):
        mesh_loaded = df.Mesh()

        self.h5file.read(mesh_loaded, name, False)

        return mesh_loaded

    def load_field(self, field_name, t):
        index = np.where(t_array)[0][0]
        name = field_name + str(index)

        f_loaded = df.Function(functionspace)
        self.h5file.read(f_loaded, name, False)

        return f_loaded

    def close(self):
        self.h5file.close()
