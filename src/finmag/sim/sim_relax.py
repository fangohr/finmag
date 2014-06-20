import functools
import logging

from finmag.util.consts import ONE_DEGREE_PER_NS
from finmag.util.helpers import compute_dmdt

log = logging.getLogger(name="finmag")

ONE_DEGREE_PER_NS = 17453292.5  # in rad/s


def relax(sim, save_vtk_snapshot_as=None, save_restart_data_as=None,
          stopping_dmdt=1.0, dt_limit=1e-10, dmdt_increased_counter_limit=10):
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
    log.debug("Relaxation parameters: stopping_dmdt={} (degrees per "
              "nanosecond), dt_limit={}, dmdt_increased_counter_limit={}"
              .format(stopping_dmdt, dt_limit, dmdt_increased_counter_limit))

    dt0 = 1e-14  # Initial dt.

    # Relaxation information is stored in a dictionary. This data is lost if
    # another relaxation takes place after this one.
    sim.relaxation = {}

    # Define relaxation event.
    def dt_interval(self):
        """
        This function calculates the interval given the internal state of an
        event instance.

        The first time this method is called, variables are initialised to
        starting values. This method then calculates the timestep for the next
        iteration.

        """
        # If the initial variables have not been defined, we must be in the
        # first iteration. Initialise the variables now.
        if 'dt' not in sim.relaxation.keys():
            sim.relaxation['dt'] = dt0
            sim.relaxation['dt_increment_multi'] = 1.5
            sim.relaxation['dt_limit'] = dt_limit
            sim.relaxation['dmdts'] = []
            sim.relaxation['stopping_dmdt'] = stopping_dmdt * ONE_DEGREE_PER_NS

        # Otherwise, calculate new dt value.
        else:
            if sim.relaxation['dt'] < sim.relaxation['dt_limit'] /\
               sim.relaxation['dt_increment_multi']:
                if len(sim.relaxation['dmdts']) >= 2 and\
                   sim.relaxation['dmdts'][-1][1] <\
                   sim.relaxation['dmdts'][-2][1]:
                    sim.relaxation['dt'] *=\
                        sim.relaxation['dt_increment_multi']
            else:
                sim.relaxation['dt'] = sim.relaxation['dt_limit']

        return sim.relaxation['dt']

    def trigger():
        """
        This function calculates dm/dt and the energy of the magnetisation 'm',
        and determines whether or not the magnetisation in this simulation is
        relaxed.

        The above result is differenced with the result from the previous
        iteration (if any). The integration is stopped when dm/dt falls below
        'stopping_dmdt', or when the system is deemed to be diverging (that is
        to say, when dm/dt and energy increase simultaneously more than
        'dmdt_increased_counter_limit' times).

        """
        # If this is the first iteration, define initial variables.
        if 'energies' not in sim.relaxation.keys():
            sim.relaxation['energies'] = []
            sim.relaxation['dmdt_increased_counter'] = 0

        # Otherwise, find dm/dt and energy and compare.
        else:
            sim.relaxation['dmdts'].append([sim.t, compute_dmdt(sim.relaxation['last_time'], sim.relaxation['last_m'], sim.t, sim.m)])
            sim.relaxation['energies'].append(sim.total_energy())

            # Continue iterating if dm/dt is not low enough.
            if sim.relaxation['dmdts'][-1][1] >\
               sim.relaxation['stopping_dmdt']:
                log.debug("At t={:.3g}, last_dmdt={:.3g} * stopping_dmdt, "
                          "next dt={:.3g}."
                          .format(sim.t, sim.relaxation['dmdts'][-1][1] /
                                  sim.relaxation['stopping_dmdt'],
                                  sim.relaxation['dt']))

            # If dm/dt and energy have both increased, start checking for
            # non-convergence.
            if len(sim.relaxation['dmdts']) >= 2:
                if (sim.relaxation['dmdts'][-1][1] >
                    sim.relaxation['dmdts'][-2][1] and
                    sim.relaxation['energies'][-1] >
                    sim.relaxation['energies'][-2]):

                    # Since dm/dt has increased, we increment the counter.
                    sim.relaxation['dmdt_increased_counter'] += 1
                    log.debug("dmdt {} times larger than last time "
                              "(counting {}/{})."
                              .format(sim.relaxation['dmdts'][-1][1] /
                                      sim.relaxation['dmdts'][-2][1],
                                      sim.relaxation['dmdt_increased_counter'],
                                      sim.relaxation['dmdt_increased_counter_limit']))

                    # The counter is too high, so we leave.
                    if sim.relaxation['dmdt_increased_counter'] >=\
                       sim.relaxation['dmdt_increased_counter_limit']:
                        log.warning("Stopping time integration after dmdt "
                                    "increased {} times without a decrease in "
                                    "energy (which indicates that something "
                                    "might be wrong)."
                                    .format(sim.relaxation['dmdt_increased_counter_limit']))
                        return False

            # Since dm/dt is low enough, stop the integration.
            if sim.relaxation['dmdts'][-1][1] <=\
               sim.relaxation['stopping_dmdt']:
                log.debug("Stopping integration at t={:.3g}, with dmdt={:.3g},"
                          " smaller than threshold={:.3g}."
                          .format(sim.t, sim.relaxation['dmdts'][-1][1],
                                  float(sim.relaxation['stopping_dmdt'])))
                return False

        # Update values for the next integration.
        sim.relaxation['last_m'] = sim.m.copy()
        sim.relaxation['last_time'] = sim.t

    dt_interval = functools.partial(dt_interval, sim)
    sim.scheduler.add(trigger, every=dt_interval,
                      after=1e-14 if sim.t < 1e-14 else sim.t)

    sim.scheduler.run(sim.integrator, sim.callbacks_at_scheduler_events)
    sim.integrator.reinit()
    sim.set_m(sim.m)
    log.info("Relaxation finished at time t = {:.2g}.".format(sim.t))

    # Save a vtk snapshot and/or restart data of the relaxed state.
    if save_vtk_snapshot_as is not None:
        sim.save_vtk(save_vtk_snapshot_as, overwrite=True)
    if save_restart_data_as is not None:
        sim.save_restart_data(save_restart_data_as)
