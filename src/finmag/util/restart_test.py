import datetime
import numpy as np
import finmag
import restart

def test_can_read_restart_file():
	sim = finmag.example.barmini()
	sim.run_until(1e-21)  # create integrator
	restart.save_restart_data(sim)
	data = restart.load_restart_data(sim)

	assert data['simname'] == sim.name
	assert data['stats'] == sim.integrator.stats()
	assert np.all(data['m'] == sim.integrator.llg.m)
	#writing and reading the data should take less than 10 seconds
	assert datetime.datetime.now() - data['datetime'] < datetime.timedelta(0, 10)

