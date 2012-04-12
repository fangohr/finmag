import time

class Timings(object):
    def __init__(self):
        """Key of data dictionary is the name for timings, values are a tuple which is (n,t,st) where
        n is the number of calls, and t the cumulative time it took, and st the status ('finished',STARTTIME)"""
        self.reset()

        self._last = None #remember which activity we started to measure last.

    def reset(self):
        self.data = {}
        self.creationtime = time.time()

    def start(self,name):
        if name in self.data.keys():
            assert self.data[name][2]=='finished',"Seems a measurement for '%s' has started already?" % name
            self.data[name][2]=time.time()
        else:
            self.data[name]=[0,0.,time.time()]
        self._last = name

    def stop(self,name):
        assert name in self.data.keys(),"name '%s' not known. Known values: %s" % self.data.keys()
        assert self.data[name][2] != 'finished',"No measurement started for name '%s'" % name
        timetaken = time.time()-self.data[name][2]
        #print 'time taken for name "%s"=%g s' % (name,timetaken)
        self.data[name][0] += 1
        self.data[name][1] += timetaken
        self.data[name][2] = 'finished'
        self._last = None

    def stoplast(self):
        """Stop the last measurement at this point."""
        assert self._last != None
        self.stop(self._last)

    def startnext(self,name):
        """Will stop whatever measurement has been started most recently, and start the 
        next one with name 'name'."""
        if self._last:
            self.stop(self._last)
        self.start(name)

    def getncalls(self,name):
        return self.data[name][0]

    def gettime(self,name):
        return self.data[name][1]

    def report_str(self,n=10):
        """Lists the n items that took the longest time to execute."""
        msg = "Timings summary, longest items first:\n"
        #print in descending order of time taken
        sorted_keys = sorted(self.data.keys(),key=lambda x:self.data[x][1],reverse=True)
        for name in sorted_keys:
            if self.data[name][0]>0:
                msg += "%25s:%6d calls took %10.4fs (%8.6fs per call)\n" % (name[0:25],
                                                                           self.getncalls(name),
                                                                           self.gettime(name),
                                                                           self.gettime(name)\
                                                                               /float(self.getncalls(name))
                                                                           )
            else:
                msg = "Timings %s: none completed\n" % name
        recorded_sum= self.recorded_sum()
        walltime = time.time()-self.creationtime
        msg+="Wall time: %.4gs (sum of time recorded: %gs=%5.1f%%)\n" % \
            (walltime,recorded_sum,recorded_sum/walltime*100.)

        return msg

    def __str__(self):
        return self.report_str()

    def recorded_sum(self):
        return sum([ self.data[name][1] for name in self.data.keys()])

timings=Timings()

if __name__=="__main__":
    #create global object that can be shared        
    t=Timings()
    for x in xrange(20000):
        t.start("test-one")
        t.stop("test-one")
    print t

