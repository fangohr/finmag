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


#sim.load_mesh("bar.nmesh.h5",
#              [("Py", mat_Py)],
#              unit_length=SI(1e-9,"m"))

sim.set_m([1,0,1])

dt = SI(5e-12, "s")

######
# After ten time steps, plot the energy density
# from z=0nm to z=100nm through the center of the body.
######

sim.advance_time(dt*10)
#E1 = sim.get_subfield("E_exch_Py")
#E2 = sim.get_subfield("E_demag_Py")
#E3 = sim.probe_subfield_siv("E_exch_Py", [1e-9, 1e-9, 1e-9])

f = open("nmag_exch_Edensity.txt", "w")
f2 = open("nmag_demag_Edensity.txt", "w")
for i in range(100):
    f.write("%g " % sim.probe_subfield_siv("E_exch_Py", [15e-9, 15e-9, 1e-9*i]))
    f2.write("%g " % sim.probe_subfield_siv("E_demag_Py", [15e-9, 15e-9, 1e-9*i]))
f.close()
f2.close()
