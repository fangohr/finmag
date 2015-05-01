import dolfin as df
import numpy as np
import json
from collections import OrderedDict
from savingdata import SavingData, LoadingData

h5filename = 'file.h5'
npzfilename = 'file.npz'
jsonfilename = 'file.json'
mesh = df.UnitSquareMesh(10, 10)
functionspace = df.VectorFunctionSpace(mesh, 'CG', 1, 3)
f = df.Function(functionspace)
t_array = np.linspace(0, 1e-9, 5)


def test_save_data():
    sd = SavingData(h5filename, npzfilename, jsonfilename, functionspace)

    sd.save_mesh()

    for i in range(len(t_array)):
        f.assign(df.Constant((t_array[i], 0, 0)))
    
        sd.save_field(f, 'f', t_array[i])

    sd.close()


def test_load_data():
    ld = LoadingData(h5filename, npzfilename, jsonfilename, functionspace)

    mesh_loaded = ld.load_mesh()

    for t in t_array:
        f_loaded = ld.load_field('f', t)

        f.assign(df.Constant((t, 0, 0)))
        assert np.all(f.vector().array() == f_loaded.vector().array())

    ld.close()


def test_load_data_with_json_data():
    ld = LoadingData(h5filename, npzfilename, jsonfilename, functionspace)

    mesh_loaded = ld.load_mesh()

    for t in t_array:
        f_loaded = ld.load_field_with_json_data('f', t)

        f.assign(df.Constant((t, 0, 0)))
        assert np.all(f.vector().array() == f_loaded.vector().array())

    ld.close()


def test_saved_json_data():
    with open(jsonfilename) as jsonfile:
        jsonData = json.load(jsonfile, object_pairs_hook=OrderedDict)
    jsonfile.close()

    # assert times from keys
    for i, t in enumerate(t_array):
        name = "f{}".format(i)
        jsonTime = jsonData['f'][name]
        assert(jsonTime == t)

    # assert times by iterating through values (as they should be ordered)
    index = 0
    for jsonTime in jsonData['f'].itervalues():
         assert(jsonTime == t_array[index])
         index += 1


