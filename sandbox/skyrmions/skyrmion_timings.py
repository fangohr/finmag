import dolfin as df
from finmag import Simulation
from finmag.energies import Exchange, DMI
output = open("timings.txt", "w")
mesh = df.Box(0, 0, 0, 30e-9, 30e-9, 3e-9, 10, 10, 1)

def run_sim(exc_jac, dmi_jac):
    sim = Simulation(mesh, Ms=8.6e5)
    sim.set_m((1, 0, 0))

    exchange = Exchange(C=1.3e-11)
    exchange.in_jacobian = exc_jac
    sim.add(exchange)

    dmi = DMI(D=4e-3)
    dmi.in_jacobian = dmi_jac
    sim.add(dmi)

    #sim.run_until(1e-9)
    sim.relax()
    output.write(sim.timings(5))

output.write("Neither in Jacobian:\n")
run_sim(exc_jac=False, dmi_jac=False)
#output.write("\nOnly DMI in Jacobian:\n")
#run_sim(exc_jac=False, dmi_jac=True) # CVODE fails with CV_TOO_MUCH_WORK
output.write("\nOnly Exchange in Jacobian:\n")
run_sim(exc_jac=True, dmi_jac=False)
output.write("\nBoth in Jacobian:\n")
run_sim(exc_jac=True, dmi_jac=True)
output.close()
print "Timings written to timings.txt"
