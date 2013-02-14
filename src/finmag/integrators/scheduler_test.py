import pytest
from events import SingleEvent, RepeatingEvent
from scheduler import Scheduler

class Counter(object):
    cnt_every = 0
    cnt_at = 0

    def inc_every(self):
        self.cnt_every += 1

    def inc_at(self):
        self.cnt_at += 1

    def reset(self):
        self.cnt_every = 0
        self.cnt_at = 0


def test_first_every_at_start():
    e = RepeatingEvent(100)
    assert e.next == 0.0

    e = RepeatingEvent(100, delay=5)
    assert e.next == 5


def test_update_next_stop_according_to_interval():
    e = RepeatingEvent(100)
    assert e.next == 0
    e.trigger(0)
    assert e.next == 100
    t0 = e.next
    e.trigger(100)
    t1 = e.next
    assert abs(t1 - t0) == 100


def test_can_attach_callback():
    c = Counter()

    assert c.cnt_every == 0
    e = RepeatingEvent(100)
    e.attach(c.inc_every)
    e.trigger(0)
    assert c.cnt_every == 1

    # alternative syntax

    c.reset()

    e = RepeatingEvent(100).call(c.inc_every)
    assert c.cnt_every == 0
    e.trigger(0)
    assert c.cnt_every == 1


def test_at_with_single_value():
    c = Counter()

    assert c.cnt_at == 0
    a = SingleEvent(100)
    assert a.next == 100
    a.attach(c.inc_at)
    a.trigger(0)
    assert c.cnt_at == 0
    a.trigger(100)
    assert c.cnt_at == 1


def test_returns_None_if_no_actions_or_done():
    s = Scheduler()
    with pytest.raises(StopIteration):
        s.next()

    def bogus():
        pass

    s.add(bogus, at=1)
    assert s.next() == 1
    s.reached(1)

    with pytest.raises(StopIteration):
        s.next()


def test_scheduler():
    c = Counter()

    s = Scheduler()
    s.add(c.inc_every, every=200)
    assert c.cnt_every == 0
    assert s.next() == 0.0
    s.add(c.inc_at, at=100)
    s.reached(0.0)
    assert c.cnt_every == 1
    assert c.cnt_at == 0
    assert s.next() == 100
    s.reached(100)
    assert c.cnt_every == 1
    assert c.cnt_at == 1
    assert s.next() == 200
    s.reached(200)
    assert c.cnt_every == 2
    assert c.cnt_at == 1

    # If a later timestep is passed to 's.reached()' then actions in
    # between *won't* be triggered.
    s.reached(500)
    assert c.cnt_every == 2  # still the same as before
    assert c.cnt_at == 1  # still the same as before


def test_scheduler_every():
    c = Counter()

    s = Scheduler()
    s.add(c.inc_every, every=100, after=5)
    assert c.cnt_every == 0
    assert s.next() == 5
    s.reached(5)
    assert c.cnt_every == 1
    s.reached(100)           # shouldn't trigger any event
    assert c.cnt_every == 1  # ... thus the counter shouldn't increase
    s.reached(105)
    assert c.cnt_every == 2
    s.reached(205)
    assert c.cnt_every == 3


def test_scheduler_clear():
    c = Counter()

    s = Scheduler()
    s.add(c.inc_every, every=5)
    assert c.cnt_every == 0
    assert s.next() == 0.0
    s.reached(0)
    assert c.cnt_every == 1
    s.reached(5)
    assert c.cnt_every == 2

    # Clear the schedule and assert that nothing is supposed to happen any more
    s.clear()
    assert(s.items == [])
    assert(s.realtime_items == {})
    s.reached(10)
    assert c.cnt_every == 2  # still the same as before
    with pytest.raises(StopIteration):
        s.next()


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
    s.add(my_at_fun, at=1, at_end=True) # can trigger twice
    s.add(my_at_fun_accident, at=2, at_end=True) # 2 is also end, should trigger only once
    s.add(my_every_fun, every=1, after=1, at_end=True) # twice
    s.add(my_standalone_at_end_fun, at_end=True) # once anyways

    assert s.next() == 1
    s.reached(1)
    assert x == [1, 0, 1, 0]
    s.reached(2)
    assert x == [1, 1, 2, 0]
    s.finalise(2) 
    assert x == [2, 1, 2, 1]
