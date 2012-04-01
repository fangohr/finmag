import time
from finmag.util.timings import Timings
def test_timings():

    t=Timings()
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
    assert t.getncalls('one') == 2
    print t

    
if __name__=="__main__":
    test_timings()
