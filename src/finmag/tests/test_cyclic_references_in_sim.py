
def test_cyclic_refs_in_simulation_object():

    import finmag
    import dolfin as df
    mesh = df.UnitIntervalMesh(1)
    s = finmag.Simulation(mesh, Ms=1, unit_length=1e-9, name='simple')
    refcount = s.delete()
    # The number 4 is emperical. If it increases, we
    # have introduced an extra cyclic reference.
    assert refcount == 4
