import functools
import time


class _TimerContext(object):

    def __init__(self, timer, name, category=None):
        self.timer = timer
        self.name = name
        self.category = category

    def __enter__(self):
        self.timer.start(self.name, self.category)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.timer.stop(self.name, self.category)
        return False


class Timer(object):

    def __init__(self):
        self._active = {}
        self._totals = {}
        self._counts = {}

    def _key(self, name, category=None):
        return (category or "", name)

    def __call__(self, name, category=None):
        return _TimerContext(self, name, category)

    def start(self, name=None, category=None):
        key = self._key(name or "", category)
        self._active[key] = time.time()

    def stop(self, name=None, category=None):
        key = self._key(name or "", category)
        started = self._active.pop(key, None)
        if started is None:
            return
        duration = time.time() - started
        self._totals[key] = self._totals.get(key, 0.0) + duration
        self._counts[key] = self._counts.get(key, 0) + 1

    def reset(self):
        self._active = {}
        self._totals = {}
        self._counts = {}

    def method(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            category = None
            if args:
                category = args[0].__class__.__name__
            name = func.__name__
            self.start(name, category)
            try:
                return func(*args, **kwargs)
            finally:
                self.stop(name, category)
        return wrapper

    def report(self, n=10):
        rows = sorted(self._totals.items(), key=lambda item: item[1], reverse=True)
        lines = []
        for (category, name), total in rows[:n]:
            count = self._counts.get((category, name), 0)
            label = name if not category else "{}.{}".format(category, name)
            lines.append("{:<40} {:>12.6f}s {:>8d}".format(label, total, count))
        return "\n".join(lines)


timer = Timer()
