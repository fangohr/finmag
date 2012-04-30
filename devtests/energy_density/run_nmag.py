import nmag
from nmag import SI

mat_Py = nmag.MagMaterial(name="Py",
                          Ms=SI(0.86e6,"A/m"),
                          exchange_coupling=SI(13.0e-12, "J/m"),
                          llg_damping=0.5)

sim = nmag.Simulation("bar")

sim.load_mesh("coarse_bar.nmesh.h5",
              [("Py", mat_Py)],
              unit_length=SI(1e-9,"m"))

sim.set_m([1,0,1])

dt = SI(5e-12, "s")

sim.advance_time(dt*10)
E1 = sim.get_subfield("E_exch_Py")
E2 = sim.get_subfield("E_demag_Py")
print E2
