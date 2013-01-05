import os.path
import logging
import numpy
logger = logging.getLogger(name='finmag')


class Tablewriter(object):
    # It is recommended that the comment symbol should end with a
    # space so that there is no danger that it gets mangled up with
    # the 'time' field because some of the code below relies on them
    # being separated by some whitespace.
    comment_symbol = '# '

    def __init__(self, filename, simulation, override=False, entity_order=None):
        logger.debug("Creating DataWriter for file '%s'" % (filename))

        # formatting for columns (could in principle be customized
        # through extra arguments here)
        precision=12
        charwidth = 18
        self.float_format = "%" + str(charwidth)+'.'+str(precision)+ "g "
        self.string_format = "%" + str(charwidth) + "s "

        # entities:
        # Idea is to have a dictionary of keys where the keys
        # are the column headers in the data file.
        # the value is a tuple (a, b) where a shows the units
        # of the data and b(sim) is a function that can be called
        # with the simulation object and will retrieve the required
        # data from the simulation object.
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
        self.entities = {
            'time': {'unit': '<s>',
                        'get': lambda sim: sim.t,
                        'header': 'time'},
            'm': {'unit': '<>',
                  'get': lambda sim: sim.m_average,
                  'header': ('m_x', 'm_y', 'm_z')}
            }

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
        
        self.save_head=False

        self.sim = simulation

    def default_entity_order(self):
        keys = self.entities.keys()
        # time needs to go first
        keys.remove('time')
        return ['time'] + sorted(keys)
    
    def update_entity_order(self):
        self.entity_order = self.default_entity_order()

    def headers(self):
        """return line one and two of ndt data file as string"""
        line1 = [self.comment_symbol]
        line2 = [self.comment_symbol]
        for entityname in self.entity_order:
            colheaders = self.entities[entityname]['header']
            # colheaders can be a 3-tuple ('mx','my','mz'), say
            # or a string ('time'). Avoid iterating over string:
            if isinstance(colheaders, str):
                colheaders = [colheaders]
            for colhead in colheaders:
                line1.append(self.string_format % colhead)
                line2.append(self.string_format % \
                    self.entities[entityname]['unit'])
        return "".join(line1) + "\n" + "".join(line2) + "\n"

    def save(self):
        """Append data (spatial averages of fields) for current configuration"""
        
        if not self.save_head:
            f = open(self.filename, 'w')
            # Write header
            f.write(self.headers())
            f.close()
            self.save_head=True
        
        # open file
        with open(self.filename, 'a') as f:
            f.write(' ' * len(self.comment_symbol))  # account for comment symbol width
            for entityname in self.entity_order:
                value = self.entities[entityname]['get'](self.sim)
                if isinstance(value, numpy.ndarray):
                    if len(value) == 3:  # 3d vector
                        for i in range(3):
                            f.write(self.float_format % value[i])
                    else:
                        msg = "Can only deal with 3d-numpy arrays so far, but shape is %s" % value.shape
                        raise NotImplementedError(msg)
                elif isinstance(value, float) or isinstance(value, int):
                    f.write(self.float_format % value)
                else:
                    msg = "Can only deal with numpy arrays, float and int so far, but type is %s" % type(value)
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

        # use numpy to read remaining data
        self.data = numpy.loadtxt(self.f)
        self.f.close()

        # some consistency checks: must have as many columns as
        # headers (disregarding the comment symbol)
        assert self.data.shape[1] == len(headers) - 1

        datadic = {}
        # now wrap up data conveniently
        for i, entity in enumerate(headers[1:]):
            datadic[entity] = self.data[:, i]

        self.datadic = datadic

    def entities(self):
        """Returns list of available entities"""
        return self.datadic.keys()

    def time(self):
        """Returns list of available time steps"""
        return self.datadic['time']

    def __getitem__(self, entity):
        """Given the entity name, return the data as numpy array"""
        return self.datadic[entity]

if __name__ == "__main__":
    #create example simulation
    import finmag
    import dolfin as df
    xmin, ymin, zmin = 0, 0, 0    # one corner of cuboid
    xmax, ymax, zmax = 6, 6, 11   # other corner of cuboid
    nx, ny, nz = 3, 3, 6         # number of subdivisions (use ~2nm edgelength)
    mesh = df.Box(xmin, ymin, zmin, xmax, ymax, zmax, nx, ny, nz)
    # standard Py parameters
    sim = finmag.sim_with(mesh, Ms=0.86e6, alpha=0.5, unit_length=1e-9, A=13e-12, m_init=(1, 0, 1))
    filename = 'data.txt'
    ndt = Tablewriter(filename, sim)
    times = numpy.linspace(0, 3.0e-11, 6 + 1)
    for i, time in enumerate(times):
        print("In iteration {}, computing up to time {}".format(i, time))
        sim.run_until(time)
        ndt.save()

    # now open file for reading
    f = Tablereader(filename)
    print f.time()
    print f['m_x']
