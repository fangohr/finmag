import logging

log = logging.getLogger("finmag")


def save_ndt(sim):
    """Given the simulation object, saves one line to the ndt finalise
    (through the TableWriter object)."""
    if sim.driver == 'cvode':
        log.debug("Saving data to ndt file at t={} (sim.name={}).".format(sim.t, sim.name))
    else:
        raise NotImplementedError("Only cvode driver known.")
    sim.tablewriter.save()
