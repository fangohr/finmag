import time
from finmag.util.timings import timed, mtimed, ftimed, Timings, timings

t = Timings()

def test_by_hand():
    t.start('one')
    time.sleep(0.1)
    t.stop('one')

    assert t.getncalls('one') == 1

def test_decorated_function():

    @ftimed(t)
    def two():
        time.sleep(0.1)

    two()
    two()

    assert t.getncalls('two') == 2

def test_decorated_function_default_timer():

    @ftimed()
    def twob():
        time.sleep(0.1)

    twob()
    twob()

    assert timings.getncalls('twob') == 2

def test_decorated_method():

    class Foo(object):
        @mtimed(t)
        def three(self):
            time.sleep(0.1)

    foo = Foo()
    foo.three()
    foo.three()
    foo.three()

    assert t.getncalls('three') == 3

def test_timed_code():

    for i in range(4):
        with timed('four', timer=t):
            time.sleep(0.1)

    assert t.getncalls('four') == 4

def test_timed_code_default_timer():

    for i in range(4):
        with timed('fourb'):
            time.sleep(0.1)

    assert timings.getncalls('fourb') == 4

if __name__ == "__main__":
    test_decorated_function()
