# every function which starts with test_
# will get automatically by the test runner.

# you can have as many test functions per file as you want.

def test_skyrmion():
    TOLERANCE = 1e-6

    # Setup your problem, compute some results.
    my_result = 1e-5
    expected_result = 1e-5

    assert abs(my_result - expected_result) < TOLERANCE


