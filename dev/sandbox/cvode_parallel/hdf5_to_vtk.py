import numpy as np
import dolfin as df

df.parameters.reorder_dofs_serial = True
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#mpirun -n 2 python test.py


if __name__ == '__main__':

    mesh = df.BoxMesh(0, 0, 0, 30, 30, 100, 6, 6, 20)
    S3 = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
       
    m_init = df.Constant([1, 1, 1.0])
    m = df.interpolate(m_init, S3)
    
    file_vtk = df.File('m.pvd')
    file = df.HDF5File(m.vector().mpi_comm(),'test.h5','r')
    
    ts = np.linspace(0, 2e-10, 101)
    
    for t in ts:
        
        #sim.field.vector().set_local(sim.llg.effective_field.H_eff)
        file.read(m,'/m_%g'%t)
        
        file_vtk << m
