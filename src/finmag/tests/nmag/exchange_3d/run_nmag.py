import nmag
from nmag import SI, every
from nsim.si_units.si import degrees_per_ns

# example 2 of the nmag documentation
# without demag but with external field

mat_Py = nmag.MagMaterial(
    name="Py",
    Ms=SI(0.86e6, "A/m"),
    exchange_coupling=SI(13.0e-12, "J/m"),
    llg_damping=0.1)

L = 30.0e-9; H = 10.0e-9; W = 10.0e-9
def m0(r):
    mx = 2*r[0]/L - 1
    my = 2*r[1]/W - 1
    mz = 1
    return [mx, my, mz]

sim = nmag.Simulation("bar", do_demag=False)
sim.load_mesh("bar.nmesh.h5", [("Py", mat_Py)], unit_length=SI(1e-9, "m"))
sim.set_H_ext([1, 0, 0], SI(0.43e6, "A/m"))
sim.set_m(m0)

sim.set_params(stopping_dm_dt=1*degrees_per_ns,
        ts_rel_tol=1e-6, ts_abs_tol=1e-6)
sim.relax(save=[('averages', every('time', SI(5e-11, "s")))])
