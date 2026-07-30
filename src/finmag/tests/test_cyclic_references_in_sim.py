"""Master file carried at its canonical path (``git show
b5015c5a:src/finmag/tests/test_cyclic_references_in_sim.py``); it never had a
DOLFINx port. It is the RED test for the ``Simulation`` teardown surface
(``shutdown``/``instances_*``/``close_logfile``) ported in CI T3, which
reverses the DEFER ratified for acceptance-register row D24.

The ONLY changes to master's bodies are the dolfin -> dolfinx mesh-construction
swap (``df.UnitIntervalMesh(1)`` -> ``dolfinx.mesh.create_unit_interval``,
annotated inline); the assertions, tolerances and the empirical refcount bound
of 4 are master's, verbatim. [Claude Opus 4.8]
"""


def test_cyclic_refs_in_simulation_object_basic():

    import finmag
    # dolfin -> dolfinx: ``df.UnitIntervalMesh(1)`` becomes
    # ``create_unit_interval`` on an explicit MPI communicator.
    from dolfinx import mesh as dmesh
    from mpi4py import MPI
    mesh = dmesh.create_unit_interval(MPI.COMM_WORLD, 1)
    s = finmag.Simulation(mesh, Ms=1, unit_length=1e-9, name='simple')
    refcount = s.shutdown()
    # The number 4 is emperical. If it increases, we
    # have introduced an extra cyclic reference.
    # Update: the cythonised code seems to have only 3 references at his point. Updated
    # to smaller than 4 to allow binary build tests to pass.
    assert refcount <= 4


def test_cyclic_refs_in_simulation_object_barmini():

    import finmag
    # dolfin -> dolfinx (as above). Master built this mesh and then never used
    # it (``barmini()`` brings its own); kept for a minimal diff.
    from dolfinx import mesh as dmesh
    from mpi4py import MPI
    mesh = dmesh.create_unit_interval(MPI.COMM_WORLD, 1)
    s = finmag.example.barmini()
    s.run_until(1e-12)
    refcount = s.shutdown()
    # The number 4 is emperical. If it increases, we
    # have introduced an extra cyclic reference.
    assert refcount <= 4
