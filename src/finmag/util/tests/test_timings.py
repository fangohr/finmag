import time
import finmag.util.timings as t

default_timer = t.timings
test_timer = t.Timings()

def test_by_hand():
    test_timer.start('one')
    time.sleep(0.1)
    test_timer.stop('one')

    assert test_timer.getncalls('one') == 1

def test_decorated_function():

    @t.ftimed(test_timer)
    def two():
        time.sleep(0.1)

    two()
    two()

    assert test_timer.getncalls('two') == 2

def test_decorated_function_default_timer():

    @t.ftimed()
    def twob():
        time.sleep(0.1)

    twob()
    twob()

    assert default_timer.getncalls('twob') == 2

def test_decorated_method():

    class Foo(object):
        @t.mtimed(test_timer)
        def three(self):
            time.sleep(0.1)

    foo = Foo()
    foo.three()
    foo.three()
    foo.three()

    assert test_timer.getncalls('three') == 3

def test_decorated_method_default_timer():

    class Foo(object):
        @t.mtimed()
        def three(self):
            time.sleep(0.1)

    foo = Foo()
    foo.three()
    foo.three()
    foo.three()

    assert default_timer.getncalls('three') == 3

def test_timed_code():

    for i in range(4):
        with t.timed('four', timer=test_timer):
            time.sleep(0.1)

    assert test_timer.getncalls('four') == 4

def test_timed_code_default_timer():

    for i in range(4):
        with t.timed('fourb'):
            time.sleep(0.1)

    assert default_timer.getncalls('fourb') == 4

if __name__ == "__main__":
    test_decorated_function()
