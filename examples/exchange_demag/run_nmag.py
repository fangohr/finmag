import nmag
from nmag import SI

mat_Py = nmag.MagMaterial(name="Py",
                          Ms=SI(0.86e6,"A/m"),
                          exchange_coupling=SI(13.0e-12, "J/m"),
                          llg_damping=0.5)

sim = nmag.Simulation("bar")

sim.load_mesh("bar30_30_100.nmesh.h5",
              [("Py", mat_Py)],
              unit_length=SI(1e-9,"m"))

sim.set_m([1,0,1])

dt = SI(5e-12, "s")

for i in range(0, 61):
    sim.advance_time(dt*i)                  #compute time development

    if i % 10 == 0:                         #every 10 loop iterations,
        sim.save_data(fields="all")         #save averages and all
                                            #fields spatially resolved
    else:
        sim.save_data()                     #otherwise just save averages

    if i == 10:
        f = open("nmag_exch_Edensity.txt", "w")
        f2 = open("nmag_demag_Edensity.txt", "w")
        for i in range(100):
            f.write("%g " % sim.probe_subfield_siv("E_exch_Py", [15e-9, 15e-9, 1e-9*i]))
            f2.write("%g " % sim.probe_subfield_siv("E_demag_Py", [15e-9, 15e-9, 1e-9*i]))
        f.close()
        f2.close()

