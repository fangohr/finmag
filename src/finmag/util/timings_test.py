import finmag.util.timings as t
import time

default_timer = t.default_timer
test_timer = t.Timings()

def test_by_hand():
    test_timer.start('timings_test', 'one')
    time.sleep(0.01)
    test_timer.stop('timings_test', 'one')

    assert test_timer.calls('timings_test', 'one') == 1

def test_decorated_function():

    @t.ftimed(test_timer)
    def two():
        time.sleep(0.01)

    two()
    two()

    assert test_timer.calls('timings_test', 'two') == 2

def test_decorated_function_default_timer():

    @t.ftimed # without parantheses
    def twob():
        time.sleep(0.01)

    twob()
    twob()

    assert default_timer.calls('timings_test', 'twob') == 2

    @t.ftimed() # with parentheses
    def twoc():
        time.sleep(0.01)

    twoc()
    twoc()

    assert default_timer.calls('timings_test', 'twoc') == 2

def test_decorated_method():

    class Foo(object):
        @t.mtimed(test_timer)
        def three(self):
            time.sleep(0.01)

    foo = Foo()
    foo.three()
    foo.three()
    foo.three()

    assert test_timer.calls('Foo', 'three') == 3

def test_decorated_method_default_timer():

    class Foo(object):
        @t.mtimed() # with parentheses
        def three(self):
            time.sleep(0.01)

        @t.mtimed # without parentheses
        def threeb(self):
            time.sleep(0.01)

    foo = Foo()
    foo.three()
    foo.three()
    foo.three()
    foo.threeb()
    foo.threeb()
    foo.threeb()

    assert default_timer.calls('Foo', 'three') == 3
    assert default_timer.calls('Foo', 'threeb') == 3

def test_timed_code():

    for i in range(4):
        with t.timed('timings_test', 'four', timer=test_timer):
            time.sleep(0.01)

    assert test_timer.calls('timings_test', 'four') == 4

def test_timed_code_default_timer():

    for i in range(4):
        with t.timed('timings_test', 'fourb'):
            time.sleep(0.01)

    assert default_timer.calls('timings_test', 'fourb') == 4

def test_regression_return_values_not_affected():

    @t.ftimed
    def my_func():
        time.sleep(0.01)
        return 1

    assert my_func() == 1

    class MyClass(object):
        @t.mtimed
        def my_method(self):
            time.sleep(0.01)
            return 1

    mo = MyClass()
    assert mo.my_method() == 1

if __name__ == "__main__":
    test_by_hand()
    test_decorated_method()
    test_decorated_function()
    test_timed_code()
    print test_timer
