import pytest
from derivedevents import SingleEvent, RepeatingEvent
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
    assert e.next_time == 0.0

    e = RepeatingEvent(100, 5)
    assert e.next_time == 5


def test_update_next_stop_according_to_interval():
    e = RepeatingEvent(100)
    assert e.next_time == 0
    e.check_and_trigger(0)
    assert e.next_time == 100
    t0 = e.next_time
    e.check_and_trigger(100)
    t1 = e.next_time
    assert abs(t1 - t0) == 100


def test_can_attach_callback():
    c = Counter()

    assert c.cnt_every == 0
    e = RepeatingEvent(100)
    e.attach(c.inc_every)
    e.check_and_trigger(0)
    assert c.cnt_every == 1

    # alternative syntax

    c.reset()

    e = RepeatingEvent(100, callback=c.inc_every)
    assert c.cnt_every == 0
    e.check_and_trigger(0)
    assert c.cnt_every == 1


def test_at_with_single_value():
    c = Counter()

    assert c.cnt_at == 0
    a = SingleEvent(100)
    assert a.next_time == 100
    a.attach(c.inc_at)
    a.check_and_trigger(0)
    assert c.cnt_at == 0
    a.check_and_trigger(100)
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


def test_reached():
    c = Counter()
    s = Scheduler()
    s.add(c.inc_every, every=10)
    s.add(c.inc_at, at=20)
    assert c.cnt_every == 0
    assert c.cnt_at == 0
    assert s.next() == 0.0

    # Trigger the first couple of events
    s.reached(0); assert c.cnt_every == 1; assert c.cnt_at == 0
    s.reached(10); assert c.cnt_every == 2; assert c.cnt_at == 0

    # Call reached() with a time step that skips the next scheduled
    # one; this should *not* trigger any events!
    s.reached(30); assert c.cnt_every == 2; assert c.cnt_at == 0

    # Now call reached() with the next scheduled time step, assert
    # that it triggered the event. Then do a couple more steps for
    # sanity checks.
    s.reached(20); assert c.cnt_every == 3; assert c.cnt_at == 1
    s.reached(25); assert c.cnt_every == 3; assert c.cnt_at == 1
    s.reached(30); assert c.cnt_every == 4; assert c.cnt_at == 1

    # It is not an error to call reached() with a time step in the
    # past. However, this won't trigger any events here because the
    # RepeatingEvent knows about its next time step, and the
    # SingleEvent was already triggered above (and no event is
    # triggered twice for the same time step, unless a reset()
    # happens).
    s.reached(10); assert c.cnt_every == 4; assert c.cnt_at == 1
    s.reached(20); assert c.cnt_every == 4; assert c.cnt_at == 1


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


def test_illegal_arguments():
    def dummy_func():
        pass

    s = Scheduler()
    with pytest.raises(AssertionError):
        s.add(dummy_func, at=0, after=1)  # delays don't mix with 'at'
    with pytest.raises(AssertionError):
        s.add(dummy_func, at=1, every=2)  # can't mix 'at' with 'every'


def test_reset_with_every():
    c = Counter()
    s = Scheduler()
    s.add(c.inc_every, every=10)
    assert c.cnt_every == 0

    # Trigger a few events at their scheduled times
    s.reached(0); assert c.cnt_every == 1
    s.reached(10); assert c.cnt_every == 2
    s.reached(20); assert c.cnt_every == 3
    s.reached(30); assert c.cnt_every == 4

    # Reset time from 30 to 15 (note: in between two scheduled time
    # steps); check that no additional events were triggered and that
    # the next time step is as expected
    s.reset(15);
    assert c.cnt_every == 4
    assert s.next() == 20

    # Trigger a few more events
    s.reached(20); assert c.cnt_every == 5
    s.reached(30); assert c.cnt_every == 6

    # Reset time again, this time precisely to a scheduled time step
    # (namely, 20); again, no additional events should have been
    # triggered and the next scheduled time step should still be 20.
    s.reset(20);
    assert c.cnt_every == 6
    assert s.next() == 20

    # Trigger a few more events
    s.reached(20); assert c.cnt_every == 7
    s.reached(30); assert c.cnt_every == 8


def test_reset_with_at():
    c = Counter()
    s = Scheduler()
    s.add(c.inc_at, at=10)
    assert c.cnt_at == 0

    s.reached(0); assert c.cnt_at == 0
    s.reached(10); assert c.cnt_at == 1
    s.reached(20); assert c.cnt_at == 1
    s.reached(30); assert c.cnt_at == 1

    # Events that already happened are not triggered again ...
    s.reached(10); assert c.cnt_at == 1

    # ... unless we reset the scheduler first
    s.reset(2)
    s.reached(10); assert c.cnt_at == 2

    # Resetting to a time *after* the scheduled time will result in
    # the event not being triggered again, even if we tell the
    # scheduler that the time step was reached.
    s.reset(30)
    s.reached(10); assert c.cnt_at == 2
