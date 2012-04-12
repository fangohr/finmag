import nmag
from nmag import SI
import math

"""
periodic spinwaves example straight from nmag's documentation
http://nmag.soton.ac.uk/nmag/0.2/manual/html/example_periodic_spinwaves/doc.html
minus the periodic part...

"""

# define magnetic material
Py = nmag.MagMaterial(name="Py",
                      Ms=SI(1e6,"A/m"),
                      exchange_coupling=SI(13.0e-12, "J/m"),
                      llg_damping = SI(0.02,"")
                      )

#create simulation object
sim = nmag.Simulation("spinwaves", do_demag=False)

# load mesh
sim.load_mesh("film.nmesh", [("film", Py)], unit_length=SI(1e-9,"m") )

# function to set the magnetisation 
def perturbed_magnetisation(pos):
    x,y,z = pos
    newx = x*1e9
    newy = y*1e9
    if 8<newx<14 and -3<newy<3:
        # the magnetisation is twisted a bit 
        return [1.0, 5.*(math.cos(math.pi*((newx-11)/6.)))**3 *\
                        (math.cos(math.pi*(newy/6.)))**3, 0.0]
    else:
        return [1.0, 0.0, 0.0]

# set initial magnetisation
sim.set_m(perturbed_magnetisation)

# let the system relax generating spin waves
s = SI("s")
from nsim.when import every, at
sim.relax(save=[('averages', every('time', 0.05e-12*s) | at('convergence'))],
          do=[('exit', at('time', 10e-12*s))])
