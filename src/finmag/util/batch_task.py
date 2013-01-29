import os
import time
import numpy as np
import multiprocessing
from multiprocessing import Process

class TaskState(object):
    def __init__(self,taskname):
        self.taskname=taskname
        self.state={}
        self.load()
    
    def load(self):
        if not os.path.exists(self.taskname):
            return
        f=open(self.taskname,'r')
        data=f.read()
        f.close()
        
        for line in data.splitlines():
            k,v=line.split(':')
            self.state[k.strip()]=v
    
    def save_state(self):
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
    def __init__(self,fun,processes=None,taskname='task',checking_time=10):
        self.fun=fun
        self.tasks=[{}]
        self.parameters=[]
        self.current_directory=os.getcwd()
        if processes:
            self.processes=processes
        else:
            self.processes=multiprocessing.cpu_count()
            
        self.ts=TaskState(taskname+'.txt')
        self.checking_time=checking_time
        self.dims=[]
        
        self.process_res=[]
        
    def add_parameters(self,name,values):
        new_tasks=[]
        self.parameters.append(name)
        self.dims.append(len(values))
        
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
        #p.daemon = True
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
        
        while job_id<len(self.utasks):
            for n, p in enumerate(jobs):
                if not p or not p.is_alive():
                    a_id=active_task_ids[n]
                    if a_id>=0:
                        self.ts.update_state(self.utasks[a_id], True)
                        self.ts.save_state()
                    jobs[n]=self.run_single(self.utasks[job_id])
                    active_task_ids[n]=job_id
                    job_id+=1
                    if job_id>=len(self.utasks):
                        break
            
            time.sleep(self.checking_time)
    
    def post_process(self,fun):
        for task in self.tasks:
            dirname=self.generate_directory(task)
            os.chdir(dirname)
            self.process_res.append(fun(**task))
            os.chdir(self.current_directory)
    
    def get_res(self,key,value):
        res=[]
        par=[]
        for i,task in enumerate(self.tasks):
            if key in task and task[key]==value:
                res.append(self.process_res[i])
                tmp_task=dict(task)
                del tmp_task[key]
                par.append(tmp_task.values()[0])
                if len(tmp_task.values())>1:
                    raise RuntimeError('Only support two variables here!')
        
        return np.array(par),np.array(res)

    
def task(p1,p2):
    res='p1='+str(p1)+'  p2='+str(p2)
    with open('res.txt','w') as f:
        f.write(res)


if __name__=="__main__":
    tasks=BatchTasks(task,4)
    tasks.add_parameters('p1',['a','b','c'])
    tasks.add_parameters('p2',range(1,5))
    #tasks.start()
    