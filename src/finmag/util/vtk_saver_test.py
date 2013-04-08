import os
import pytest
import dolfin as df
import numpy as np
from vtk_saver import VTKSaver
from finmag.util.helpers import assert_number_of_files


class TestVTKSaver(object):
    def setup_class(self):
        """
        Create a dummy field in various formats (numpy arrays and
        dolfin function).
        """
        mesh = df.BoxMesh(0, 0, 0, 1, 1, 1, 5, 5, 5)
        S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
        N = mesh.num_vertices()
        self.field_data = df.Function(S3)
        # The next line is a hack and not recommended for real work
        self.field_data.vector().array()[:] = np.zeros(3 * N)

    def test_constructor(self, tmpdir):
        """
        Check various methods of creating a VTKSaver object.
        """
        os.chdir(str(tmpdir))
        v1 = VTKSaver()
        v2 = VTKSaver('myfile.pvd')

    def test_file_extension_is_correct(self, tmpdir):
        """
        Check that only filenames with extension '.pvd' are accepted.
        """
        os.chdir(str(tmpdir))
        VTKSaver("myfile.pvd")
        with pytest.raises(ValueError): VTKSaver("myfile.vtk")
        with pytest.raises(ValueError): VTKSaver("myfile.vtu")
        with pytest.raises(ValueError): VTKSaver("myfile.txt")

    def test_existing_files_are_deleted_if_requested(self, tmpdir):
        """

        """
        os.chdir(str(tmpdir))

        # Create a few (empty) dummy .pvd and .vtu files
        with open("myfile.pvd", 'w'): pass
        with open("myfile000001.vtu", 'w'): pass
        with open("myfile000002.vtu", 'w'): pass
        with open("myfile000003.vtu", 'w'): pass
        with open("myfile000004.vtu", 'w'): pass

        # Trying to re-open an existing .pvd file should raise an error:
        with pytest.raises(IOError):
            VTKSaver("myfile.pvd")

        # Unless 'overwrite' is set to True, in which case the .vtu
        # files should be deleted:
        v = VTKSaver("myfile.pvd", overwrite=True)
        v.save_field(self.field_data, t=0)  # just to create a single .pvd and .vtu file
        assert_number_of_files("myfile.pvd", 1)
        assert_number_of_files("myfile*.vtu", 1)

    def test_save_field(self, tmpdir):
        """
        Check that calling save_field with the field data given in
        various format works and creates the expected .vtu files.
        """
        os.chdir(str(tmpdir))

        v = VTKSaver("myfile.pvd")

        v.save_field(self.field_data, t=0.0)
        assert_number_of_files("myfile.pvd", 1)
        assert_number_of_files("myfile*.vtu", 1)

        v.save_field(self.field_data, t=1e-12)
        assert_number_of_files("myfile.pvd", 1)
        assert_number_of_files("myfile*.vtu", 2)

        v.save_field(self.field_data, t=3e-12)
        v.save_field(self.field_data, t=8e-12)
        assert_number_of_files("myfile.pvd", 1)
        assert_number_of_files("myfile*.vtu", 4)

    def test_saving_to_file_with_different_name(self, tmpdir):
        os.chdir(str(tmpdir))

        v = VTKSaver("myfile1.pvd")
        v.save_field(self.field_data, t=0.0)
        v.save_field(self.field_data, t=0.1)
        v.open("myfile2.pvd")
        v.save_field(self.field_data, t=0.0)
        v.save_field(self.field_data, t=0.1)
        v.save_field(self.field_data, t=0.2)

        assert_number_of_files("myfile1.pvd", 1)
        assert_number_of_files("myfile1*.vtu", 2)

        assert_number_of_files("myfile2.pvd", 1)
        assert_number_of_files("myfile2*.vtu", 3)
