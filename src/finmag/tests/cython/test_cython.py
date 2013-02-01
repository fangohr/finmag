""" This is a bit of an odd test: it checks that a work-around for a
particular cython problem is still working. More meant to document
a solution rather than test cython, but can't do much harm.

HF 1 Feb 2013
"""

import subprocess


#Cython complains about this:
#  f_callable_normalised = vector_valued_function(lambda (x,y,z): (a*x, b*y, c*z), S3, normalise=True)

# try to reproduce with a simple example

def myfunc(callable):
    print callable((0, 1, 2))
    print callable((10, 10, 10))


def cython_test_code():

    a, b, c = 1, 1, 1
    x, y, z = 1, 1, 1

    # cython complains about this
    #myfunc(lambda (x, y, z): (a * x, b * y, c * z))

    # but this one works:
    myfunc(lambda t: (a * t[0], b * t[1], c * t[1]))

    # and so does this:
    def myf(tup):
        x, y, z = tup
        return (a * x, b * y, c * z)

    myfunc(myf)


def test_cython_compiles_this_file():
    cmd = "cython {}.py".format('test_cython')
    print("about to execute {}".format(cmd))
    subprocess.check_call(cmd, shell=True)

if __name__ == '__main__':
    test_cython_compiles_this_file()
