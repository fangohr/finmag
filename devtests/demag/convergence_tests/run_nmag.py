import nmag, sys
from nmag import SI

try:
	meshfile = sys.argv[1]
	datafile = sys.argv[2]
except IndexError:
	print 'Usage: nmsim %s meshfile outputdatafile' % sys.argv[0]
	sys.exit(1)

#create simulation object
sim = nmag.Simulation()

# define magnetic material
Py = nmag.MagMaterial(name = 'Py',
                      Ms = SI(1.0, 'A/m'),
                      exchange_coupling = SI(13.0e-12, 'J/m'))

# load mesh
sim.load_mesh(meshfile,
              [('sphere', Py)],
              unit_length = SI(1e-9, 'm'))

# set initial magnetisation
sim.set_m([1,0,0])

# set external field
sim.set_H_ext([0,0,0], SI('A/m'))

# Save and display data in a variety of ways
sim.save_data(fields='all') # save all fields spatially resolved
                            # together with average data

import numpy as np
Hd = sim.get_subfield('H_demag')

Hdx = Hd[:,0]
N = len(Hdx)
exct = -1./3*np.ones(N)
stddev = np.sqrt(1./N*sum((Hdx- exct)**2))

f = open(datafile, 'a')
f.write('%s %s %s\n' % (str(np.average(Hdx)), str(max(Hdx)), str(stddev)))
f.close()
