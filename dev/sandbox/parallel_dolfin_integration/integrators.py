# This library includes some functions for integrating arrays.

def euler(Tn, dTndt, tStep):
    """
    Performs Euler integration to obtain T_{n+1}.

    Arguments:
       Tn: Array-like representing Temperature at time t_n.
       dTndt: Array-like representing dT/dt at time t_n.
       tStep: Float determining the time to step over.

    Returns T_{n+1} as an array-like.
    """
    return Tn + dTndt * tStep
