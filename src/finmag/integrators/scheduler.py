from operator import attrgetter

class Every(object):
    def __init__(self, interval, start=0.0):
        self.next_step = start
        self.interval = interval
        self.callback = None

    def attach(self, callback):
        self.callback = callback
        return self

    call = attach

    def fire(self):
        self.next_step += self.interval

        if self.callback:
            self.callback()

class At(object):
    def __init__(self, time):
        self.next_step = time
        self.callback = None

    def attach(self, callback):
        self.callback = callback
        return self

    call = attach

    def fire(self):
        self.next_step = None
        if self.callback:
            self.callback()

class Scheduler(object):
    def __init__(self):
        self.items = []

    def add(self, at_or_every):
        self.items.append(at_or_every)

    def next_step(self):
        next_steps = [i.next_step for i in self.items if i.next_step is not None]
        if len(next_steps) == 0:
            return None
        return min(next_steps)

    def reached(self, time):
        for item in self.items:
            if item.next_step == time:
                item.fire()
