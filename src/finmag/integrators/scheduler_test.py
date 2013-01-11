from scheduler import Every, At, Scheduler

def test_first_every_at_start():
    e = Every(100)
    assert e.next_step == 0.0

    e = Every(100, 5)
    assert e.next_step == 5

def test_update_next_stop_according_to_interval():
    e = Every(100)
    e.update()

    t0 = e.next_step
    e.update()
    t1 = e.next_step

    assert abs(t1 - t0) == 100

def test_can_attach_callback():
    a = [0]
    def my_fun():
        a[0] += 1

    assert a[0] == 0
    e = Every(100)
    e.attach(my_fun)
    e.fire(0)
    assert a[0] == 1

    # alternative syntax

    b = [0]
    def my_funb():
        b[0] += 1

    e = Every(100).call(my_funb)
    assert b[0] == 0
    e.fire(0)
    assert b[0] == 1

def test_at():
    x = [0]
    def my_fun():
        x[0] += 1

    a = At(100)
    assert a.next_step == 100
    a.attach(my_fun)
    a.fire(0)
    assert x[0] == 1

def test_returns_None_if_no_actions_or_done():
    s = Scheduler()
    assert s.next_step() == None

    s.add(None, at=1)
    assert s.next_step() == 1
    s.reached(1)
    assert s.next_step() == None

def test_scheduler():
    x = [0, 0]
    def my_fun():
        x[0] += 1
    def my_funb():
        x[1] += 1

    s = Scheduler()
    s.add(my_fun, every=200)
    assert x[0] == 0
    assert s.next_step() == 0.0
    s.add(my_funb, at=100)
    s.reached(0.0)
    assert x[0] == 1
    assert x[1] == 0
    assert s.next_step() == 100
    s.reached(100)
    assert x[0] == 1
    assert x[1] == 1
    assert s.next_step() == 200
    s.reached(200)
    assert x[0] == 2
    assert x[1] == 1

def test_regression_not_more_than_once_per_time():
    x = [0, 0, 0, 0]
    def my_at_fun():
        x[0] += 1
    def my_at_fun_accident():
        x[1] += 1
    def my_every_fun():
        x[2] += 1
    def my_standalone_at_end_fun():
        x[3] += 1

    s = Scheduler()
    s.add(my_at_fun, at=1, at_end=True) # can fire twice
    s.add(my_at_fun_accident, at=2, at_end=True) # 2 is also end, should fire only once
    s.add(my_every_fun, every=1, after=1, at_end=True) # twice
    s.add(my_standalone_at_end_fun, at_end=True) # once anyways

    assert s.next_step() == 1
    s.reached(1)
    assert x == [1, 0, 1, 0]
    s.reached(2)
    assert x == [1, 1, 2, 0]
    s.finalise(2) 
    assert x == [2, 1, 2, 1]
