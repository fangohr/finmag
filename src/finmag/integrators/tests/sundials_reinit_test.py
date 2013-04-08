# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Hans Fangohr (copied template from Dmitri for
#  integrator tests)

"""The purpose of this file is to test whether the reinit() function
is working correctly. This should tell Sundials' cvode integrator that
it should expect some drastic change in the equation we integrate. While
the values of the dynamic degrees of freedom should remain the same, we
want to warn the integrator that the right hand side may change quickly.

In the magnetic context, this could be a sudden change in the applied field
which will propagate to the effective field, and thus result in a sudden
change in the torque from the precession term (and others).

After receiving the re-init signal, sundials will forget the history of
the integration, and start to explore the equations again to determine
appropriate time steps.


It would be nice to show that this actually happens, but I have not found
a good example yet. This file can be run directly, and it shows that
using re-init can actually decrease the total number of function evaluations.

It also shows that sundials resets the counters for the evalutaions of
the right hand side. So for now, this is the only firm criterion used
here for testing.

Hans, 30 June 2012
"""

from finmag.tests.jacobean.domain_wall_cobalt import setup_domain_wall_cobalt, \
                                                     domain_wall_error
from finmag.integrators.llg_integrator import llg_integrator
from datetime import datetime

NODE_COUNT = 100
END_TIME1 = 0.1e-10
END_TIME2 = END_TIME1 + 0.1e-10


def run_test(backend, method, mode='onego', nsteps=40000):
    llg = setup_domain_wall_cobalt(node_count=NODE_COUNT)
    integrator = llg_integrator(llg, llg.m, backend, method=method, nsteps=nsteps)
    t = datetime.now()

    if mode == 'onego':
        END_TIME = END_TIME1 + END_TIME2
    elif mode == 'twogoes' or mode == 'twogoesreinit':
        END_TIME = END_TIME1
    else:
        raise ValueError("Can only understand 'onego', 'twogoes', twogoesreinit'.")

    integrator.advance_time(END_TIME)
    dt = datetime.now() - t
    print "backend=%s, method=%s: elapsed time=%s, n_rhs_evals=%s, error=%g" % (
            backend,
            method,
            dt,
            integrator.n_rhs_evals,
            domain_wall_error(integrator.m, NODE_COUNT))
    if mode == 'onego':
        return integrator

    if mode == 'twogoesreinit':
        #check that rhs counter goes back to zero
        print "re-initialising"
        integrator.reinit()
        assert integrator.n_rhs_evals == 0
    else:
        print "Not re-initialising"

    integrator.advance_time(END_TIME2)
    print "backend=%s, method=%s: elapsed time=%s, n_rhs_evals=%s, error=%g" % (
            backend,
            method,
            dt,
            integrator.n_rhs_evals,
            domain_wall_error(integrator.m, NODE_COUNT))
    print("second call to integrator.n_rhs_evals ={}".format(integrator.n_rhs_evals))
    return integrator



def test_reinit_resets_num_rhs_eval_counter():
    int = run_test("sundials", "bdf_diag", mode='twogoesreinit')
    int = run_test("sundials", "adams", mode='twogoesreinit')
    int = run_test("sundials", "adams", mode='twogoesreinit')
    return

if __name__ == '__main__':
    #the actual test
    test_reinit_resets_num_rhs_eval_counter()

    print "Demo how nhs_rhs_evals changes with and without reinit"
    int = run_test("sundials", "bdf_diag", mode='twogoes')
    int = run_test("sundials", "bdf_gmres_no_prec", mode='twogoesreinit')
    int = run_test("sundials", "adams", mode='onego')


#def not_used_here_test_scipy():
#    return run_test("scipy", "bdf")

# def test_scipy_bdf(self):
#         self.run_test("scipy", "bdf")

#     def test_scipy_adams(self):
#         self.run_test("scipy", "adams")

#     def test_sundials_adams(self):
#         self.run_test("sundials", "bdf_diag")

#     def test_sundials_bdf_diag(self):
#         self.run_test("sundials", "adams")

#     def test_sundials_bdf_gmres_no_prec(self):
#         self.run_test("sundials", "bdf_gmres_no_prec")

#     def test_sundials_bdf_gmres_prec_id(self):
#         self.run_test("sundials", "bdf_gmres_prec_id")
