import os
import re
import glob
import time
import logging
import dolfin as df

log = logging.getLogger("finmag")

class VTKSaver(object):
    def __init__(self, filename=None, overwrite=False):
        self.filename = filename
        self.f = None
        self.counter = 0
        if filename != None:
            self.open(filename, overwrite)

    def open(self, filename, overwrite=False):
        ext = os.path.splitext(filename)[1]
        if ext != '.pvd':
            raise ValueError(
                "File extension for vtk snapshot file must be '.pvd', "
                "but got: '{}'".format(ext))
        self.filename = filename
        self.basename = re.sub('\.pvd$', '', self.filename)

        if os.path.exists(self.filename):
            if overwrite:
                log.warning(
                    "Removing file '{}' and all associated .vtu files "
                    "(because force_overwrite=True).".format(self.filename))
                os.remove(self.filename)
                for f in glob.glob(self.basename + "*.vtu"):
                    os.remove(f)
            else:
                raise IOError(
                    "Aborting snapshot creation. File already exists and "
                    "would overwritten: '{}' (use force_overwrite=True if "
                    "this is what you want)".format(self.filename))


        # Open the file here so that it stays open during all calls to
        # save(), otherwise consecutive calls will overwrite previously
        # written data.
        self.f = df.File(self.filename, "compressed")

    def save_field(self, field_data, t):
        """
        Save the given field data to the .pvd file associated with
        this VTKSaver.


        *Arguments*

        field_data:  dolfin.Function

            The data to be saved.

        t:  float

            The time step with which the data is associated

        """
        self.counter += 1
        t0 = time.time()
        self.f << field_data
        t1 = time.time()
        log.debug("Saved field at t={} to file '{}' (snapshot #{}; saving took "
                  "{:.3g} seconds).".format(t, self.filename, self.counter, t1 - t0))
