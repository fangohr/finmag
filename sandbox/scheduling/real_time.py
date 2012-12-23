from time import gmtime, strftime, sleep
from apscheduler.scheduler import Scheduler

# Start the scheduler
sched = Scheduler()
sched.start()

def job_function():
    print "Hello! It is now {}.".format(strftime("%H:%M:%S", gmtime()))

# Schedule job_function to be called every 5 seconds
sched.add_interval_job(job_function, seconds=5)

print "This script will run for quarter a minute and a job is scheduled to execute every 5 seconds."
print "Imagine this is progress information of a long running simulation to show every 10 minutes or so."
# sleep for a minute so that script doesn't exit
sleep(15)
