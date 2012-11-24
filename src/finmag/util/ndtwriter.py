import os.path
import logging
logger = logging.getLogger(name='finmag')


class ndtWriter(object):

    datatoinclude = {
        'time': ('<s>', lambda sim: sim.time),
        'm': ('<A/m>', lambda sim: sim.m_average)
    }

    #def headers():

    def __init__(self, filename, simulation):
        self.filename = filename
        self.sim = simulation
        # if file exists, cowardly stop
        if os.path.exists(filename):
            msg = "File %s exists already; cowardly stopping" % filename
            raise RuntimeError(msg)
        self.f = open(self.filename, 'w')

    def append(self):
        for entity in sorted(datatoinclude.keys()):
            value = datatoinclude[entity][1]()
            if type(value) == numpy.ndarray:
                if len(value) == 3:  # 3d vector
                    for i in range(3):
                        f.write("%g\t" % value[i]) 
            f.write('\n')
