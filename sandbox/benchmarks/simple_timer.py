import time

class SimpleTimer():
    """
    A simple timer which can be used as a context manager.

    Usage example:

        benchmark = SimpleTimer()
        with benchmark:
            do_stuff()
        print benchmark.elapsed

    A SimpleTimer instance can be reused.

    """
    def __init__(self):
        self.start = 0.0
        self.stop = 0.0

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        self.stop = time.time()
        self.elapsed = self.stop - self.start
