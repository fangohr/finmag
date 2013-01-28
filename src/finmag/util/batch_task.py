import os
import time
import multiprocessing
from multiprocessing import Process

class TaskState(object):
    def __init__(self,taskname):
        self.taskname=taskname
        self.state={}
        self.load()
        print self.state
    
    def load(self):
        if not os.path.exists(self.taskname):
            return
        f=open(self.taskname,'r')
        data=f.read()
        f.close()
        
        for line in data.splitlines():
            k,v=line.split(':')
            self.state[k.strip()]=v
    
    def save(self):
        f=open(self.taskname,'w')
        for k in self.state:
            f.write('%s : %s\n'%(k,self.state[k]))
        f.close()
        
    def update_state(self,k,v):
        key=self.dict2str(k)
        if v:
            self.state[key]='Done!'
        else:
            self.state[key]='Waiting!'
        self.save()
        
    def dict2str(self,d):
        res=[]
        for k in d:
            res.append(k)
            res.append(str(d[k]))
        return '_'.join(res)
        
    
    def done(self,k):
        key=self.dict2str(k)
        if key in self.state:
            if 'Done' in self.state[key]:
                return True
        else:
            self.update_state(k,False)
        return False


class BatchTasks(object):
    def __init__(self,fun,processes=None,taskname='task'):
        self.fun=fun
        self.tasks=[{}]
        self.parameters=[]
        self.current_directory=os.getcwd()
        if processes:
            self.processes=processes
        else:
            self.processes=multiprocessing.cpu_count()
            
        self.ts=TaskState(taskname+'.txt')
        
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
        jobs=[]
        job_id=0
        active_task_ids=[]
        for _ in range(self.processes):
            jobs.append(None)
            active_task_ids.append(-1)
        
        self.utasks=[]
        for task in self.tasks:
            if not self.ts.done(task):
                self.utasks.append(task)
        print self.utasks
        
        while job_id<len(self.utasks):
            for n, p in enumerate(jobs):
                if not p or not p.is_alive():
                    a_id=active_task_ids[n]
                    if a_id>=0:
                        self.ts.update_state(self.utasks[a_id], True)
                    jobs[n]=self.run_single(self.utasks[job_id])
                    active_task_ids[n]=job_id
                    job_id+=1
                    if job_id>=len(self.utasks):
                        break
            time.sleep(10)
        

def task(p1,p2):
    res='p1='+str(p1)+'  p2='+str(p2)
    with open('res.txt','w') as f:
        f.write(res)

if __name__=="__main__":
    tasks=BatchTasks(task,4)
    tasks.add_parameters('p1',['a','b','c'])
    tasks.add_parameters('p2',range(0,10,2))
    tasks.start()
    