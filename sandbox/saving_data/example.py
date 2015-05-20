import savingdata as sd
import dolfin as df
import numpy as np

# Define mesh, functionspace, and functions.
mesh = df.UnitSquareMesh(5, 5)
fs = df.VectorFunctionSpace(mesh, 'CG', 1, 3)
f1 = df.Function(fs)
f2 = df.Function(fs)
f3 = df.Function(fs)

f1.assign(df.Constant((1, 1, 1)))
f2.assign(df.Constant((2, 2, 2)))
f3.assign(df.Constant((3, 1, 6)))

# Filenames. At the moment both npz and json files are implemented.
h5filename = 'hdf5file.h5'
jsonfilename = 'jsonfile.json'
npzfilename = 'npzfile.npz'

# SAVING DATA.
sdata = sd.SavingData(h5filename, npzfilename, jsonfilename, fs)

sdata.save_mesh(name='mesh')

sdata.save_field(f1, 'm', t=0)
sdata.save_field(f2, 'm', t=1e-12)
sdata.save_field(f3, 'm', t=2e-12)

# Temporarily disable close. Problems on virtual machine.
#sdata.close()


# LOADING DATA.
ldata = sd.LoadingData(h5filename, npzfilename, jsonfilename, fs)

meshl = ldata.load_mesh(name='mesh')

f1l = ldata.load_field(field_name='m', t=0)
f2l = ldata.load_field(field_name='m', t=1e-12)
f3l = ldata.load_field(field_name='m', t=2e-12)

f1lj = ldata.load_field_with_json_data(field_name='m', t=0)
f2lj = ldata.load_field_with_json_data(field_name='m', t=1e-12)
f3lj = ldata.load_field_with_json_data(field_name='m', t=2e-12)

# Temporarily disable closing. Problems on virtual machine.
# ldata.close()

# ASSERTIONS. Is the saved data the same as loaded.
assert np.all(mesh.coordinates() == meshl.coordinates())

assert np.all(f1l.vector().array() == f1.vector().array())
assert np.all(f2l.vector().array() == f2.vector().array())
assert np.all(f3l.vector().array() == f3.vector().array())

assert np.all(f1lj.vector().array() == f1.vector().array())
assert np.all(f2lj.vector().array() == f2.vector().array())
assert np.all(f3lj.vector().array() == f3.vector().array())






