import os
import logging
import types
import numpy as np
from glob import glob
from types import TupleType, StringType
from aeon import mtimed
logger = logging.getLogger(name='finmag')


class Tablewriter(object):
    # It is recommended that the comment symbol should end with a
    # space so that there is no danger that it gets mangled up with
    # the 'time' field because some of the code below relies on them
    # being separated by some whitespace.
    comment_symbol = '# '

    def __init__(self, filename, simulation, override=False, entity_order=None, entities=None):
        logger.debug("Creating DataWriter for file '%s'" % (filename))

        # formatting for columns (could in principle be customized
        # through extra arguments here)
        precision = 12
        charwidth = 18
        self.float_format = "%" + str(charwidth)+'.'+str(precision)+ "g "
        self.string_format = "%" + str(charwidth) + "s "

        # save_head records whether the headings (name and units)
        # have been saved already
        self.save_head = False

        # entities:
        # Idea is to have a dictionary of keys where the keys
        # are reference names for the entities and
        # the value is another dictionary, which has keys 'unit', 'get' and 'header':
        # 'get' is the a function that takes a simulation object as the argument
        # and returns the data to be saved.
        #
        # No doubt this can be done neater, more general, etc.
        # For example, it would be desirable if we could get ALL
        # the fields from the simulation object, i.e. demag, exchange,
        # anisotropy and also the corresponding energies.
        #
        # Ideally this would have the flexiblity to realise when we have
        # two different anisotropies in the simulation, and provide both of
        # these. It may be that we need create a 'fieldname' that the user
        # can provide when creating interactions which summarises what the
        # field is about, and which can be used as a useful column header
        # here for the ndt file.

        if entities is None:
            
            self._entities = {}
            
            self.add_entity('time', {'unit': '<s>',
	                                 'get': lambda sim: sim.t,
	                                 'header': 'time'})
            self.add_entity('m', {'unit': '<>',
	                              'get': lambda sim: sim.m_average,
	                              'header': ('m_x', 'm_y', 'm_z')})
	
	        # add time integrator dummy tokens than return NAN as we haven't got 
	        # the integrator yet (or may never create one).
            self.add_entity( 'steps',  {
	            'unit': '<1>',
	            #'get': lambda sim: sim.integrator.stats()['nsteps'],
	            'get': lambda sim: np.NAN,
	            'header': 'steps'})
            
            self.add_entity('last_step_dt', {
	            'unit': '<1>',
	            #'get': lambda sim: sim.integrator.stats()['hlast'],
	            'get': lambda sim: np.NAN,
	            'header': 'last_step_dt'})
            
            self.add_entity('dmdt', {
	            'unit': '<A/ms>',
	            #'get': lambda sim: sim.dmdt_max,
	            'get': lambda sim: np.array([np.NAN, np.NAN, np.NAN]),
	            'header': ('dmdt_x', 'dmdt_y', 'dmdt_z')})

        else:
            self._entities = entities
	
        self.filename = filename
        self.sim = simulation
        # in what order to write data
        if entity_order:
            self.entity_order = entity_order
        else:
            self.entity_order = self.default_entity_order()

        # if file exists, cowardly stop
        if os.path.exists(filename) and not override:
            msg = "File %s exists already; cowardly stopping" % filename
            raise RuntimeError(msg)


    def add_entity(self, name, dic):
        """
        Add an entity to be saved to this ndt file at the next data saving instance. The
        arguments are:

           name  : a reference name for this entity (used to order the entities in the ndt file)

           dic : a dictionary containing data for the header lines and a function to retrieve the data.

        Examples:

        For the time entity, we have

            name = 'time'
        
            dic =  {'unit': '<s>',
                     'get': lambda sim: sim.t,
                     'header': 'time'},

        For the magnetisation entity, we have

            name = 'm'
    
            dic = {'unit': '<>',
                  'get': lambda sim: sim.m_average,
                  'header': ('m_x', 'm_y', 'm_z')}
        """
        if self.save_head:
            raise RuntimeError("Attempt to add entity '{}'->'{}' to ndt file {}" + \
                               "after file has been created -- this is impossible".\
                               format(name, dic, self.filename))

        assert name not in self._entities.keys(), \
                              "Attempt to add a second '{}' to entities for {}".\
                              format(name, self.filename)

        # check that right keywords are given
        entity_descr = "entity '{}' -> '{}'".format(name, dic)
        assert 'header' in dic, "Missing 'header' in " + entity_descr
        assert 'unit' in dic, "Missing 'unit' in " + entity_descr
        assert 'get' in dic, "Missing 'get' in " + entity_descr

        self._entities[name] = dic
        self.update_entity_order()


    def modify_entity_get_method(self, name, new_get_method):
        """Allows changing the get method. Is used for integrators at the moment: we register
        dummy get methods when the tablewriter file is created, and then updated those if and
        when an integrator has been created."""
        assert name in self._entities, "Couldn't find '{}' in {}".format(name, self._entities.keys())

        logger.debug("Updating get method for {} in TableWriter(name={})".format(
            name, self.filename))
        #logger.debug("Updating get method for {} in TableWriter(name={}) old method: {}, new method: {}".format(
        #    name, self.filename, self._entities[name]['get'], new_get_method))

        self._entities[name]['get'] = new_get_method


    def delete_entity_get_method(self, name):
        """We cannot delete entities once they are created (as this would change the number of columns in the
        data file). Instead, we register a return function that returns numpy.NAN. 
        """
        assert name in self._entities, "Couldn't find '{}' in {}".format(name, self._entities.keys())

        logger.debug("'Deleting' get method for {} in TableWriter(name={})".format(
            name, self.filename))

        self._entities[name]['get'] = lambda sim: np.NAN

    def default_entity_order(self):
        keys = self._entities.keys()
        # time needs to go first
        if 'time' in keys:
            keys.remove('time')
            return ['time'] + sorted(keys)
        elif 'step' in keys:
            keys.remove('step')
            return ['step'] + sorted(keys)
        else:
            return keys
        
    def update_entity_order(self):
        self.entity_order = self.default_entity_order()

    def headers(self):
        """return line one and two of ndt data file as string"""
        line1 = [self.comment_symbol]
        line2 = [self.comment_symbol]
        for entityname in self.entity_order:
            colheaders = self._entities[entityname]['header']
            # colheaders can be a 3-tuple ('mx','my','mz'), say
            # or a string ('time'). Avoid iterating over string:
            if isinstance(colheaders, str):
                colheaders = [colheaders]
            for colhead in colheaders:
                line1.append(self.string_format % colhead)
                line2.append(self.string_format % \
                    self._entities[entityname]['unit'])
        return "".join(line1) + "\n" + "".join(line2) + "\n"

    @mtimed
    def save(self):
        """Append data (spatial averages of fields) for current
        configuration"""

        if not self.save_head:
            f = open(self.filename, 'w')
            # Write header
            f.write(self.headers())
            f.close()
            self.save_head = True

        # open file
        with open(self.filename, 'a') as f:
            f.write(' ' * len(self.comment_symbol))  # account for comment
                                                     # symbol width
## The commented lines below are Hans' initial attempt to catch when the
## number of columns to be written changes
## but this seems to never happen. So it's not quite right.
## Also, if this was the right place to catch it, i.e. if watching
## self._entities is the critical object that shouldn't change after
## the header has been written, then we should convert this into a
## 'property' which raises an error if called for writing once the
## header lines have been written. HF, 9 June 2014.
#            if len(self._entities) == self.ncolumn_headings_written:
#                msg = "It seems number of columns to be written" + \
#                    "to {} has changed".format(self.filename)
#                msg += "from {} to {}. This is not supported.".format(
#                    self.ncolumn_headings_written, len(self.entity_order))
#                logger.error(msg)
#                raise ValueError(msg)
            for entityname in self.entity_order:
                value = self._entities[entityname]['get'](self.sim)
                if isinstance(value, np.ndarray):

                    for v in value:
                        f.write(self.float_format % v)

                elif isinstance(value, float) or isinstance(value, int):
                    f.write(self.float_format % value)
                elif isinstance(value, types.NoneType):
                    #f.write(self.string_format % value)
                    f.write(self.string_format % "nan")
                else:
                    msg = "Can only deal with numpy arrays, float and int " + \
                        "so far, but type is %s" % type(value)
                    raise NotImplementedError(msg)

            f.write('\n')


class Tablereader(object):

    # open ndt file
    def __init__(self, filename):
        self.filename = filename
        # if file exists, cowardly stop
        if not os.path.exists(filename):
            raise RuntimeError("Cannot see file '%s'" % self.filename)
        # immediatey read file
        self.reload()

    def reload(self):
        """Read Table data file"""

        try:
            self.f = open(self.filename, 'r')
        except IOError:
            raise RuntimeError("Cannot see file '%s'" % self.filename)

        line1 = self.f.readline()
        line2 = self.f.readline()
        headers = line1.split()
        units = line2.split()

        assert len(headers) == len(units)

        # use numpy to read remaining data (genfromtxt will
        # complain if there are rows with different sizes)
        try:
            self.data = np.genfromtxt(self.f)
        except ValueError:
            raise RuntimeError("Cannot load data from file '{}'." +
                               "Maybe the file was incompletely written?".
                               format(self.f))
        self.f.close()

        # Make sure we have a 2d array even if the file only contains a single line (or none)
        if self.data.ndim == 1:
            self.data = self.data[np.newaxis, :]

        # Check if the number of data columns is equal to the number of headers
        assert self.data.shape[1] == len(headers) - 1

        datadic = {}
        # now wrap up data conveniently
        for i, entity in enumerate(headers[1:]):
            datadic[entity] = self.data[:, i]

        self.datadic = datadic

    def entities(self):
        """Returns list of available entities"""
        return self.datadic.keys()

    def timesteps(self):
        """Returns list of available time steps"""
        return self.datadic['time']

    def __getitem__(self, entity):
        """
        Given the entity name, return the data as a 1D numpy array.
        If multiple entity names (separated by commas) are given
        then a 2D numpy array is returned where the columns represent
        the data for the entities.
        """
        if isinstance(entity, StringType):
            res = self.datadic[entity]
        elif isinstance(entity, TupleType):
            res = [self.datadic[e] for e in entity]
        else:
            raise TypeError("'entity' must be a string or a tuple. "
                            "Got: {} ({})".format(entity, type(entity)))
        return res


class FieldSaver(object):
    """
    Wrapper class which can incrementally save data to one file or
    multiple files (depending on the file type). Internally, this
    keeps a counter which is included in the file name if multiple
    files need to be created.

    Supported file types:

       .npy  --  Creates multiple, incrementally numbered .npy files.

    """

    cnt_pattern = '_{:06d}'

    def __init__(self, filename, overwrite=False, incremental=False):
        if not filename.endswith('.npy'):
            filename += '.npy'

        # Create any non-existing directory components
        dirname = os.path.dirname(filename)
        if dirname != '' and not os.path.exists(dirname):
            os.makedirs(dirname)

        self.filename = filename
        self.basename, self.ext = os.path.splitext(filename)
        self.incremental = incremental
        self.counter = 0

        if incremental:
            existing_files = glob(self.basename + '_*' + self.ext)
        else:
            existing_files = glob(self.filename)

        if len(existing_files) > 0:
            if overwrite == False:
                raise IOError(
                    "Will not overwrite existing file(s). Use 'overwrite=True' "
                    "if this is what you want.".format(self.basename))
            else:
                logger.debug("Overwriting {} existing file(s) "
                             "'{}*.npy'.".format(len(existing_files), self.basename))
                for f in existing_files:
                    os.remove(f)

    def save(self, data):
        """
        Save the given data (which should be a numpy array).

        """
        if self.incremental:
            cur_filename = self.basename + self.cnt_pattern.format(self.counter) + self.ext
        else:
            cur_filename = self.filename

        logger.debug("Saving field data to file '{}'.".format(cur_filename))
        np.save(cur_filename, data)
        self.counter += 1


def demo2():

    import finmag
    sim = finmag.example.barmini(name='demo2-fileio')

    sim.save_averages()

    # and write some more data
    sim.schedule("save_ndt", every=10e-12)
    sim.run_until(0.1e-9)

    # read the data

    data = Tablereader('demo2_fileio.ndt')
    for t, mx, my, mz in zip(data['time'], data['m_x'], data['m_y'], data['m_z']):
        print("t={:10g}, m = {:12}, {:12}, {:12}".format(t, mx, my, mz))


def demo1():
    #create example simulation
    import finmag
    import dolfin as df
    xmin, ymin, zmin = 0, 0, 0    # one corner of cuboid
    xmax, ymax, zmax = 6, 6, 11   # other corner of cuboid
    nx, ny, nz = 3, 3, 6         # number of subdivisions (use ~2nm edgelength)
    mesh = df.BoxMesh(xmin, ymin, zmin, xmax, ymax, zmax, nx, ny, nz)
    # standard Py parameters
    sim = finmag.sim_with(mesh, Ms=0.86e6, alpha=0.5, unit_length=1e-9, A=13e-12, m_init=(1, 0, 1))
    filename = 'data.txt'
    ndt = Tablewriter(filename, sim, override=True)
    times = np.linspace(0, 3.0e-11, 6 + 1)
    for i, time in enumerate(times):
        print("In iteration {}, computing up to time {}".format(i, time))
        sim.run_until(time)
        ndt.save()

    # now open file for reading
    f = Tablereader(filename)
    print f.timesteps()
    print f['m_x']

if __name__ == "__main__":
    print("Demo 1")
    demo1()
    print("Demo 2")
    demo2()
