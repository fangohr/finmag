from finmag.native.llg import helloWorld, StochasticHeunIntegrator

def test_file_was_built():
    helloWorld()
    shi = StochasticHeunIntegrator()
    shi.helloWorld()
