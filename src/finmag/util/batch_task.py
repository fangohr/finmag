import os
import time
import multiprocessing
from multiprocessing import Process


class BatchTasks(object):
    def __init__(self,fun,processes=None):
        self.fun=fun
        self.tasks=[{}]
        self.parameters=[]
        self.current_directory=os.getcwd()
        if not processes:
            self.processes=processes
        else:
            self.processes=multiprocessing.cpu_count()
        
    def add_parameters(self,name,values):
        new_tasks=[]
        self.parameters.append(name)
        for task in self.tasks:
            for v in values:
                t=dict(task)
                t[name]=v
                new_tasks.append(t)
            
        self.tasks=list(new_tasks)
    
    def generate_directory(self,task):
        base=self.current_directory
        for name in self.parameters:
            base=os.path.join(base,name+'_'+str(task[name]))
        
        return base
    
    def run_single(self,task):
        dirname=self.generate_directory(task)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        os.chdir(dirname)
        p=Process(target=self.fun,kwargs=task)
        p.start()
        os.chdir(self.current_directory)
        return p
            
    def start(self):
        active_jobs=[]
        job_id=0
        for i in range(self.processes):
            if i<len(self.tasks):
                p=self.run_single(self.tasks[i])
                active_jobs.append(p)
                job_id=i
        
        while job_id<len(self.tasks):
            time.sleep(30)
            for n, p in enumerate(active_jobs):
                if not p.is_alive():
                    active_jobs.pop(n)
                    job_id+=1
                    if job_id>=len(self.tasks):
                        break
                    p=self.run_single(self.tasks[job_id])
                    active_jobs.append(p)
        

def task(p1,p2):
    res='p1='+str(p1)+'  p2='+str(p2)
    with open('res.txt','w') as f:
        f.write(res)

if __name__=="__main__":
    tasks=BatchTasks(task,2)
    tasks.add_parameters('p1',['a','b','c'])
    tasks.add_parameters('p2',range(0,10,2))
    tasks.start()
    