# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

from datetime import datetime, timedelta
import math

def td_seconds(td):
    return td.microseconds * 1e-6 + td.seconds + td.days * 24 * 3600

# Note to self: using Student's t distribution here makes no sense since subsequent
# time measurements are not independent. Need to find a better way to produce confidence intervals...
STUDENT_T_095 = [1e100,
                 6.31375, 2.91999, 2.35336, 2.13185, 2.01505, 1.94318, 1.89458,
                 1.85955, 1.83311, 1.81246, 1.79588, 1.78229, 1.77093, 1.76131,
                 1.75305, 1.74588, 1.73961, 1.73406, 1.72913, 1.72472, 1.72074,
                 1.71714, 1.71387, 1.71088, 1.70814, 1.70562, 1.70329, 1.70113,
                 1.69913, 1.69726, 1.69552, 1.69389, 1.69236, 1.69092, 1.68957,
                 1.6883, 1.68709, 1.68595, 1.68488, 1.68385, 1.68288, 1.68195,
                 1.68107, 1.68023, 1.67943, 1.67866, 1.67793, 1.67722, 1.67655,
                 1.67591, 1.67528, 1.67469, 1.67412, 1.67356, 1.67303, 1.67252,
                 1.67203, 1.67155, 1.67109, 1.67065, 1.67022, 1.6698, 1.6694, 1.66901,
                 1.66864, 1.66827, 1.66792, 1.66757, 1.66724, 1.66691, 1.6666,
                 1.66629, 1.666, 1.66571, 1.66543, 1.66515, 1.66488, 1.66462, 1.66437,
                 1.66412, 1.66388, 1.66365, 1.66342, 1.6632, 1.66298, 1.66277,
                 1.66256, 1.66235, 1.66216, 1.66196, 1.66177, 1.66159, 1.6614,
                 1.66123, 1.66105, 1.66088, 1.66071, 1.66055, 1.66039, 1.66023
]

class counter:
    def __init__(self, max_time_sec=1., min_time_sec=0.3, min_error=0.01, min_groups = 3, max_groups = 10, flops_per_iter=None, bytes_per_iter=None):
        self.min_time_sec = min_time_sec
        self.max_time_sec = max_time_sec
        self.min_error_ratio = min_error
        self.start_time = self.rep_start_time = self.rep_end_time = datetime.now()
        self.min_groups = min_groups
        self.max_groups = max_groups
        self.group_times = []

        self.rep_curr = -1
        self.rep_count = 1

        self.group_size = 0
        self.group_count = 0
        self.sum_times = 0
        self.sum_times_sq = 0
        self.flops_per_iter = flops_per_iter
        self.bytes_per_iter = bytes_per_iter

    def next(self):
        self.rep_curr += 1
        if  self.rep_curr < self.rep_count:
            return True
        return self.advance_rep()

    def advance_rep(self):
        self.rep_end_time = datetime.now()

        diff = td_seconds(self.rep_end_time - self.rep_start_time)
        if not self.group_count:
            # still need to find how many times to run the test until it takes at least max_time_sec/MAX_GROUPS seconds
            if diff < self.min_time_sec / self.max_groups:
                self.rep_count *= 2
                self.rep_curr = 0
                self.rep_start_time = self.rep_end_time
                return True
            else:
                self.group_size = self.rep_count

        self.add_time(diff)
        self.rep_curr = 0
        self.rep_start_time = self.rep_end_time
        if self.group_count < self.min_groups or td_seconds(self.rep_start_time - self.start_time) < self.min_time_sec:
            return True
        if td_seconds(self.rep_start_time - self.start_time) > self.max_time_sec:
            return False
        return self.confidence_time_sec() / self.mean_time_sec() > self.min_error_ratio

    def add_time(self, time):
        self.sum_times += time
        self.sum_times_sq += time * time
        self.group_count += 1
        self.group_times.append(time)

    def mean_time_sec(self):
        return self.sum_times / self.group_size / self.group_count

    def calls_per_sec(self, factor):
        return 1.*factor/self.mean_time_sec()

    def confidence_time_sec(self):
        if self.group_count < len(STUDENT_T_095):
            cutoff =  STUDENT_T_095[self.group_count]
        else:
            cutoff = 1.64485
        s2 = (self.sum_times_sq - self.sum_times**2/ self.group_count) / (self.group_count - 1)
        conf = cutoff * math.sqrt(s2 / self.group_count)
        return conf / self.group_size

    def group_times_str(self, fmt, mult):
        return " ".join(fmt % (mult*x/self.group_size) for x in self.group_times)

    def flops_str(self):
        if not self.flops_per_iter:
            return ""
        mean = self.mean_time_sec()
        return ", %.2f Gflops" % (self.flops_per_iter/mean/1e9)

    def bytes_str(self):
        if not self.bytes_per_iter:
            return ""
        mean = self.mean_time_sec()
        return ", %.2f GB/sec" % (self.bytes_per_iter/mean/1e9)

    def __str__(self):
        mean = self.mean_time_sec()
        conf = self.confidence_time_sec()
        if mean > 1e-3:
            return "%.1f calls/sec, %.1f ms/call%s%s (%.1f%% error), [%s] (%d per group)" % (1/mean, mean*1e3, self.flops_str(), self.bytes_str(), conf/mean*100, self.group_times_str("%.1f", 1e3), self.group_size)
        else:
            return "%.0f calls/sec, %.1f us/call%s%s (%.1f%% error), [%s] (%d per group)" % (1/mean, mean*1e6, self.flops_str(), self.bytes_str(), conf/mean*100, self.group_times_str("%.1f", 1e6), self.group_size)

def time_str(time_sec):
    if type(time_sec) is timedelta:
        time_sec = time_sec.total_seconds()
    if time_sec >= 1.:
        return "%.1f s" % (time_sec,)
    if time_sec >= 1e-3:
        return "%.0fms" % (time_sec*1e3,)
    return "%.0fus" % (time_sec*1e6,)

if __name__ == "__main__":
    c = counter()
    while c.next():
        pass
    print c
