import finmag

def test_restart_same_simulation():

    sim = finmag.example.barmini()
    sim.run_until(10e-12)

    # <markdowncell>

    # To be able to restart the simulation from a particular point, we need to save the magnetisation at that time before:

    # <codecell>

    sim.save_restart_data()

    # <markdowncell>

    # We can see from the message that the filename ``barmini-restart.npz`` has been chosen. This is the *canonical* filename, composed of
    # 
    # - the simulation name, and
    # - the ``-restart`` and 
    # - the default extension ``.npz`` for multiple numpy arrays saved as a zipped file
    # 
    # For completeness the simulation name is:

    # <codecell>

    print(sim.name)

    # <markdowncell>

    # Let us also save the magnetisation at this point in time.

    # <codecell>

    m_10em12 = sim.m

    # <markdowncell>

    # We can also choose any filename we like (although we need to stick to the ``.npz`` extension), for example

    # <codecell>

    sim.save_restart_data(filename="my-special-state.npz")

    # <markdowncell>

    # And show the average component values for future reference

    # <codecell>

    print("t=%s, <m>=%s" % (sim.t, sim.m_average))

    # <markdowncell>

    # Then carry on with the time integration:

    # <codecell>

    sim.run_until(100e-12)
    print("t=%s, <m>=%s" % (sim.t, sim.m_average))

    assert sim.t == 100e-12
    # <markdowncell>

    # We know imagine that we need to restart this run, or create another simulation that continues at the point of t=1e-12 where we have saved our restart snapshot:

    # <markdowncell>

    # ### Restart 

    # <markdowncell>

    # Imagine we need to go back to t=10e-12 and the corresponding magnetisation configuration. We can use:

    # <codecell>

    sim.restart()
    assert sim.t == 10e-12
    # <markdowncell>

    # If the ``restart`` method is not given any filename, it will look for the canonical restart name of its simulation object.
    # 
    # And just to convince us:

    # <codecell>

    print("time = %s " % sim.t)
    print("<m> = %s" % sim.m_average)
    assert sim.t == 10e-12
    assert (sim.m == m_10em12).all  # check that this identical to before saving

    # integrate a little so that we change time and status
    sim.run_until(20e-12)
    # <markdowncell>

    # If we want to restart from a different configuration (i.e. not from the canonical filename, we need to provide a restart file name):

    # <codecell>

    sim.restart('my-special-state.npz')
    print("time = %s " % sim.t)
    print("<m> = %s" % sim.m_average)
    assert sim.t == 10e-12
    assert (sim.m == m_10em12).all  # check that this identical to before saving


    sim.run_until(24e-12)
    assert sim.t == 24e-12
    # <markdowncell>

    # If we want to use the same magnetisation, but change the point in time at which we start the integration, we can use the optional ``t0`` parameter:

    # <codecell>

    sim.restart('my-special-state.npz', t0=0.42e-12)
    print("time = %s " % sim.t)
    print("<m> = %s" % sim.m_average)
    assert sim.t == 0.42e-12
    assert (sim.m == m_10em12).all  # check that this identical to before saving

    # <codecell>

    print("t=%s, <m>=%s" % (sim.t, sim.m_average))

    # <markdowncell>

    # ## Creating a new simulation from saved restart file

    # <markdowncell>

    # To create a new simulation that starts from a saved configurtion, we need to create the simulation object (and we have to use exactly the same mesh -- there is no check for this at the moment), and can then use the restart method as before:

    # <codecell>

def test_restart_new_simulation():
    import finmag
    sim2 = finmag.example.barmini()
    sim2.restart('my-special-state.npz')
    # <codecell>

    print("t=%s, <m>=%s" % (sim2.t, sim2.m_average))
    assert sim2.t == 10e-12
