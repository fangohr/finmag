# HF: April 2014: I am just creating this test file. I am not sure
# whether we have no other tests for the effictive fields - maybe
# they are in a different file.


# Testing new 'get_interaction_list' function:

import dolfin
import finmag

def test_get_interaction_list(): 
    # has bar mini example Demag and Exchange?
    s = finmag.example.barmini()
    lst = s.get_interaction_list()
    assert 'Exchange' in lst
    assert 'Demag' in lst
    assert len(lst) == 2

    # Let's remove one and ceck again
    s.remove_interaction('Exchange')
    assert s.get_interaction_list() == ['Demag']

    # test simulation with no interaction
    s2 = finmag.sim_with(
        mesh=dolfin.IntervalMesh(10, 0, 1),
        m_init=(1, 0, 0), Ms=1, 
        demag_solver=None, unit_length=1e-8)
    assert s2.get_interaction_list() == []


    
