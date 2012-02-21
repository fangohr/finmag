# This script will run all the unit tests in the current directory
# The unit test files must have the form *_tests.py
import unittest, os, xmlrunner, sys

if __name__ == "__main__":
    tests_module_dir = os.path.dirname(os.path.abspath(__file__))
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().discover(".", pattern="*_tests.py"))
    result = xmlrunner.XMLTestRunner(output=os.path.join(tests_module_dir, "../../../test-reports/junit")).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
