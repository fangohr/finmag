import logging
import numpy as np

log = logging.getLogger(name="finmag")

EPSILON = 1e-15
ONE_DEGREE_PER_NS = 17453292.5  # in rad/s


class Relaxation(object):
    """
    Monitors the relaxation of the magnetisation over time.

    """
    def __init__(self, stopping_dmdt=ONE_DEGREE_PER_NS, dmdt_increased_counter_limit=500, dt_limit=1e-10):
        self.dt = 1e-14
        self.dt_increment_multi = 1.5
        self.dt_limit = dt_limit

        self.stopping_dmdt = stopping_dmdt
        self.dmdt_increased_counter = 0
        self.dmdt_increased_counter_limit = dmdt_increased_counter_limit
        self.dmdts = [] # list of (t, max_dmdt) tuples

        self.last_m = None
        self.last_t = None

        # to communicate with scheduler
        self.next_step = 0.0
        self.stop_simulation = False
        self.at_end = False

    def check_interrupt_relaxation(self):
        """
        This is a backup plan in case relaxation can't be reached by normal means.
        Monitors if dmdt increases too ofen.

        """
        if len(self.dmdts) >= 2:
            if self.dmdts[-1][1] > self.dmdts[-2][1]:
                self.dmdt_increased_counter += 1
                log.debug("dmdt {} times larger than last time (counting {}/{}).".format(
                    self.dmdts[-1][1] / self.dmdts[-2][1],
                    self.dmdt_increased_counter,
                    self.dmdt_increased_counter_limit))

        if self.dmdt_increased_counter >= self.dmdt_increased_counter_limit:
            log.warning("Stopping time integration after dmdt increased {} times.".format(
                self.dmdt_increased_counter_limit))
            self.next_step = None
            self.stop_simulation = True

    def fire(self, t):
        assert abs(t - self.sim.t) < EPSILON
        if (self.last_t != None) and abs(self.last_t - t) < EPSILON:
            return
        if self.stop_simulation:
            log.error("Time integration continued even though relaxation has been reached.")

        t = self.sim.t
        m = self.sim.m.copy()

        if self.last_m != None:
            dmdt = compute_dmdt(self.last_t, self.last_m, t, m)
            self.dmdts.append((t, dmdt))

            if dmdt > self.stopping_dmdt:
                if self.dt < self.dt_limit / self.dt_increment_multi:
                    if len(self.dmdts) >= 2 and dmdt < self.dmdts[-2][1]:
                        self.dt *= self.dt_increment_multi
                else:
                    self.dt = self.dt_limit

                log.debug("At t={:.3g}, last_dmdt={:.3g} * stopping_dmdt, next dt={:.3g}.".format(
                    t, dmdt / self.stopping_dmdt, self.dt))
                self.check_interrupt_relaxation()
            else:
                log.debug("Stopping integration at t={:.3g}, with dmdt={:.3g}, smaller than threshold={:.3g}.".format(
                    t, dmdt, float(self.stopping_dmdt)))
                self.stop_simulation = True # hoping this gets noticed by the Scheduler

        self.last_t = t
        self.last_m = m
        self.next_step += self.dt


def compute_dmdt(t0, m0, t1, m1):
    """
    Returns the maximum of the L2 norm of dm/dt.

    Arguments:
        t0, t1: two points in time (floats)
        m0, m1: the magnetisation at t0, resp. t1 (np.arrays of shape 3*n)

    """
    dm = (m1 - m0).reshape((3, -1))
    max_dm = np.max(np.sqrt(np.sum(dm**2, axis=0))) # max of L2-norm
    dt = abs(t1 - t0)
    max_dmdt = max_dm / dt
    return max_dmdt
