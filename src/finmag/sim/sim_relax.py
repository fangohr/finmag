
import logging
from finmag.scheduler import derivedevents

log = logging.getLogger(name="finmag")

ONE_DEGREE_PER_NS = 17453292.5  # in rad/s

def relax(sim, save_vtk_snapshot_as=None, save_restart_data_as=None, stopping_dmdt=1.0,
          dt_limit=1e-10, dmdt_increased_counter_limit=10):
    """
    Run the simulation until the magnetisation has relaxed.

    This means the magnetisation reaches a state where its change over time
    at each node is smaller than the threshold `stopping_dm_dt` (which
    should be given in multiples of degree/nanosecond).

    If `save_vtk_snapshot_as` and/or `restart_restart_data_as` are
    specified, a vtk snapshot and/or restart data is saved to a
    file with the given name. This can also be achieved using the
    scheduler but provides a slightly more convenient mechanism.
    Note that any previously existing files with the same name
    will be automatically overwritten!

    """
    if not hasattr(sim, "integrator"):
        sim.create_integrator()
    log.info("Simulation will run until relaxation of the magnetisation.")
    log.debug("Relaxation parameters: stopping_dmdt={} (degrees per nanosecond), "
              "dt_limit={}, dmdt_increased_counter_limit={}".format(
                          stopping_dmdt, dt_limit, dmdt_increased_counter_limit))

    if hasattr(sim, "relaxation"):
        del(sim.relaxation)

    sim.relaxation = derivedevents.RelaxationEvent(sim, stopping_dmdt*ONE_DEGREE_PER_NS, dmdt_increased_counter_limit, dt_limit)
    sim.scheduler._add(sim.relaxation)

    sim.scheduler.run(sim.integrator, sim.callbacks_at_scheduler_events)
    sim.integrator.reinit()
    sim.set_m(sim.m)
    log.info("Relaxation finished at time t = {:.2g}.".format(sim.t))

    sim.scheduler._remove(sim.relaxation)
    del(sim.relaxation.sim) # help the garbage collection by avoiding circular reference

    # Save a vtk snapshot and/or restart data of the relaxed state.
    if save_vtk_snapshot_as is not None:
        sim.save_vtk(save_vtk_snapshot_as, overwrite=True)
    if save_restart_data_as is not None:
        sim.save_restart_data(save_restart_data_as)
