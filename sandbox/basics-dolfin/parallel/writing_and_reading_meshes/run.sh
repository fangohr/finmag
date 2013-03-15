#!/bin/bash

echo "========================================================================"
echo "Writing the same mesh using different numbers of processors."
echo "========================================================================"
python write_mesh_serial.py
mpirun -n 1 python write_mesh.py
mpirun -n 2 python write_mesh.py
mpirun -n 4 python write_mesh.py
mpirun -n 8 python write_mesh.py

echo "========================================================================"
echo "Trying to read various mesh files using a different number of processes."
echo "It seems that problems mainly (only?) occur when reading the file that"
echo "Was written in parallel with many (e.g. 8) processes."
echo "========================================================================"
python read_mesh_serial.py meshfile_serial.h5
python read_mesh_serial.py meshfile_08.h5
mpirun -n 2 python read_mesh.py meshfile_01.h5
mpirun -n 2 python read_mesh.py meshfile_08.h5
mpirun -n 8 python read_mesh.py meshfile_01.h5
mpirun -n 8 python read_mesh.py meshfile_08.h5
