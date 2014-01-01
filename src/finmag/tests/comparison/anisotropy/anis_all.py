import conftest as test

from test_anis_magpar import test_against_magpar
from test_anis_nmag import test_against_nmag
from test_anis_oommf import test_against_oommf

if __name__ == "__main__":
    finmag = test.setup(K2=0)

    test_against_magpar(finmag)
    test_against_oommf(finmag)

    finmag = test.setup()
    test_against_nmag(finmag)
    test.teardown(finmag)

