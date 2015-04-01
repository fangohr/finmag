
def test_cyclic_refs_in_simulation_object_basic():

    import finmag
    import dolfin as df
    mesh = df.UnitIntervalMesh(1)
    s = finmag.Simulation(mesh, Ms=1, unit_length=1e-9, name='simple')
    refcount = s.shutdown()
    # The number 4 is emperical. If it increases, we
    # have introduced an extra cyclic reference.
    # Update: the cythonised code seems to have only 3 references at his point. Updated
    # to smaller than 4 to allow binary build tests to pass.
    assert refcount <= 4

def test_cyclic_refs_in_simulation_object_barmini():

    import finmag
    import dolfin as df
    mesh = df.UnitIntervalMesh(1)
    s = finmag.example.barmini()
    s.run_until(1e-12)
    refcount = s.shutdown()
    # The number 4 is emperical. If it increases, we
    # have introduced an extra cyclic reference.
    assert refcount <= 4
