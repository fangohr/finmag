import time
import dolfin as df
from finmag.util.timings import Timings
from finmag import sim_with


def test_timings():
    t = Timings()
    print t
    t.start('one')
    time.sleep(0.2)
    t.stop('one')
    assert t.getncalls('one') == 1
    assert (t.gettime('one') - 0.2) < 0.05
    print t
    t.start('one')
    time.sleep(0.05)
    t.stop('one')
    t.start('two')
    t.startnext('three')
    time.sleep(0.1)
    t.startnext('four')
    time.sleep(0.1)
    t.startnext('five')
    time.sleep(0.1)
    t.stop('five')

    assert t.getncalls('one') == 2
    print t


if __name__ == "__main__":
    test_timings()
