import finmag.util.timings as t

default_timer = t.timings
test_timer = t.Timings()

def test_by_hand():
    test_timer.start('one')
    pass
    test_timer.stop('one')

    assert test_timer.getncalls('one') == 1

def test_decorated_function():

    @t.ftimed(test_timer)
    def two():
        pass

    two()
    two()

    assert test_timer.getncalls('two') == 2

def test_decorated_function_default_timer():

    @t.ftimed # without parantheses
    def twob():
        pass

    twob()
    twob()

    assert default_timer.getncalls('twob') == 2

    @t.ftimed() # with parentheses
    def twoc():
        pass

    twoc()
    twoc()

    assert default_timer.getncalls('twoc') == 2

def test_decorated_method():

    class Foo(object):
        @t.mtimed(test_timer)
        def three(self):
            pass

    foo = Foo()
    foo.three()
    foo.three()
    foo.three()

    assert test_timer.getncalls('three') == 3

def test_decorated_method_default_timer():

    class Foo(object):
        @t.mtimed() # with parentheses
        def three(self):
            pass

        @t.mtimed # without parentheses
        def threeb(self):
            pass

    foo = Foo()
    foo.three()
    foo.three()
    foo.three()
    foo.threeb()
    foo.threeb()
    foo.threeb()

    assert default_timer.getncalls('three') == 3
    assert default_timer.getncalls('threeb') == 3

def test_timed_code():

    for i in range(4):
        with t.timed('four', timer=test_timer):
            pass

    assert test_timer.getncalls('four') == 4

def test_timed_code_default_timer():

    for i in range(4):
        with t.timed('fourb'):
            pass

    assert default_timer.getncalls('fourb') == 4

if __name__ == "__main__":
    test_decorated_function()
