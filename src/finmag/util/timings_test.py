import finmag.util.timings as t
import time

default_timer = t.default_timer
test_timer = t.Timings()
_zZ = lambda: time.sleep(1e-3)

def test_by_hand():
    test_timer.start('one', 'timings_test')
    _zZ()
    test_timer.stop('one', 'timings_test')

    assert test_timer.calls('one', 'timings_test') == 1

def test_by_hand_without_groupname():
    test_timer.start('oneb')
    _zZ()
    test_timer.stop('oneb')

    assert test_timer.calls('oneb') == 1

def test_decorated_function():

    @t.ftimed(test_timer)
    def two():
        _zZ()

    two()
    two()

    assert test_timer.calls('two', 'timings_test') == 2

def test_decorated_function_default_timer():

    @t.ftimed # without parantheses
    def twob():
        _zZ()

    twob()
    twob()

    assert default_timer.calls('twob', 'timings_test') == 2

    @t.ftimed() # with parentheses
    def twoc():
        _zZ()

    twoc()
    twoc()

    assert default_timer.calls('twoc', 'timings_test') == 2

def test_decorated_method():

    class Foo(object):
        @t.mtimed(test_timer)
        def three(self):
            _zZ()

    foo = Foo()
    foo.three()
    foo.three()
    foo.three()

    assert test_timer.calls('three', 'Foo') == 3

def test_decorated_method_default_timer():

    class Foo(object):
        @t.mtimed() # with parentheses
        def three(self):
            _zZ()

        @t.mtimed # without parentheses
        def threeb(self):
            _zZ()

    foo = Foo()
    foo.three()
    foo.three()
    foo.three()
    foo.threeb()
    foo.threeb()
    foo.threeb()

    assert default_timer.calls('three', 'Foo') == 3
    assert default_timer.calls('three', 'Foo') == 3

def test_timed_code():

    for i in range(4):
        with t.timed('four', 'timings_test', timer=test_timer):
            _zZ()

    assert test_timer.calls('four', 'timings_test') == 4

def test_timed_code_without_group():

    for i in range(4):
        with t.timed('fourb', timer=test_timer):
            _zZ()

    assert test_timer.calls('fourb') == 4

def test_timed_code_default_timer():

    for i in range(4):
        with t.timed('fourc', 'timings_test'):
            _zZ()

    assert default_timer.calls('fourc', 'timings_test') == 4

def test_regression_return_values_not_affected():

    @t.ftimed
    def my_func():
        _zZ()
        return 1

    assert my_func() == 1

    class MyClass(object):
        @t.mtimed
        def my_method(self):
            _zZ()
            return 1

    mo = MyClass()
    assert mo.my_method() == 1

if __name__ == "__main__":
    test_by_hand()
    test_decorated_method()
    test_decorated_function()
    test_timed_code()
    print test_timer
