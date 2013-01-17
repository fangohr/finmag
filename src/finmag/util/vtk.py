import os
import re
import glob
import time
import logging
import dolfin as df
from finmag.util.timings import mtimed

log = logging.getLogger(name="finmag")


class VTK(object):
    """
    Can save dolfin functions to VTK files and allows for sequential snapshots.

    This object can be used in two modes:

    1. To save a single snapshot.

    Pass a ``filename`` to the constructor. Should that file exist, it will not
    be overwritten unless ``force_overwrite`` is set to True.

    2. To save a series of snapshots.

    In that case, don't pass in a ``filename``, but only a ``prefix``. The
    filename will then be built using the ``prefix``, the current number of
    the snapshot, and the simulation time at which is was taken.

    """
    def __init__(self, filename="", directory="", force_overwrite=False, prefix=""):
        """
        Force the object into one of the two modes described in the class documentation.

        If `filename` is empty, a default filename will be generated
        based on a sequentially increasing counter and the current
        time of the simulation. A user-defined string can be
        inserted into the generated filename by passing `prefix`. The prefix
        will be ignored if a filename is passed.

        Note that `filename` is also allowed to contain directory
        components (for example filename='snapshots/foo.pvd'), which
        are simply appended to `directory`. However, if `filename`
        contains an absolute path then the value of `directory` is
        ignored. If a file with the same filename already exists, the
        method will abort unless `force_overwrite` is True, in which
        case the existing .pvd and all associated .vtu files are
        deleted before saving the snapshot.

        If `directory` is non-empty then the file will be saved in the
        specified directory.

        All directory components present in either `directory` or
        `filename` are created if they do not already exist.

        """
        self.filename = filename
        self.directory = directory
        self.force_overwrite = force_overwrite
        self.prefix = prefix
        self.counter = 1

    @mtimed
    def save(self, dolfin_function, t):
        """
        Save the ``dolfin_function`` to a .pvd file (in VTK format) which can
        later be inspected using Paraview, for example.

        """
        filename = self.filename
        if filename == "":
            prefix_insert = "" if self.prefix == "" else self.prefix + "_"
            filename = "{}{}_{:.3f}ns.pvd".format(prefix_insert,
                self.vtk_snapshot_no, t * 1e9)

        ext = os.path.splitext(filename)[1]
        if ext != '.pvd':
            raise ValueError("File extension for vtk snapshot file must be "
                             "'.pvd', but got: '{}'".format(ext))
        if os.path.isabs(filename) and self.directory != "":
            log.warning("Ignoring 'directory' argument (value given: '{}') "
                        "because 'filename' contains an absolute path: "
                        "'{}'".format(self.directory, filename))

        output_file = os.path.join(self.directory, filename)
        if os.path.exists(output_file):
            if self.force_overwrite:
                log.warning(
                    "Removing file '{}' and all associated .vtu files "
                    "(because force_overwrite=True).".format(output_file))
                os.remove(output_file)
                basename = re.sub('\.pvd$', '', output_file)
                for f in glob.glob(basename + "*.vtu"):
                    os.remove(f)
            else:
                raise IOError(
                    "Aborting snapshot creation. File already exists and "
                    "would overwritten: '{}' (use force_overwrite=True if "
                    "this is what you want)".format(output_file))

        t0 = time.time()
        f = df.File(output_file, "compressed")
        f << dolfin_function
        t1 = time.time()
        log.info(
            "Saved snapshot at t={} to file '{}' (saving took {:.3g} seconds).".format(
                t, output_file, t1 - t0))
        self.counter += 1
