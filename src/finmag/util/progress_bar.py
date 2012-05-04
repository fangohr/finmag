import progressbar
from datetime import datetime, timedelta

DISPLAY_DELAY = timedelta(seconds=1)

class ProgressBar(object):
    def __init__(self, maximum_value):
        self.pb = progressbar.ProgressBar(maxval=maximum_value,
            widgets=[progressbar.ETA(), progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        self.display_time = datetime.now() + DISPLAY_DELAY

    def update(self, value):
        if datetime.now() > self.display_time:
            self.pb.update(value)