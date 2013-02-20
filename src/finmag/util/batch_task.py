import os
import time
import numpy as np
import multiprocessing as mp

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
        
    def update_state(self,k,v,save=True):
        key=self.dict2str(k)
        if v:
            self.state[key]='Done!'
        else:
            self.state[key]='Waiting!'
            
        if save:
            self.save_state()
        
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
            self.update_state(k,False,False)
        return False
    
    
def wrapper_function(fun,task,parameters,cwd):
    """
    something related to pickle (solution: copy_reg ?) 
    """
    
    dirname=str(cwd)
    for name in parameters:
        dirname=os.path.join(dirname,name+'_'+str(task[name]))
    
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    os.chdir(dirname)
    
    fun(**task)
            
    os.chdir(cwd)
    return
    

class BatchTasks(object):
    def __init__(self,fun,processes=None,taskname='task',waiting_time=1):
        self.fun=fun
        self.tasks=[{}]
        self.parameters=[]
        self.current_directory=os.getcwd()
        if processes:
            self.pool=mp.Pool(processes)
        else:
            self.pool=mp.Pool()
            
        self.ts=TaskState(taskname+'.txt')
        self.waiting_time=waiting_time
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
            
        def call_back(res):
            self.ts.update_state(task, True)
            
        self.pool.apply_async(wrapper_function,args=(self.fun,task,self.parameters,self.current_directory),callback=call_back)
        
            
    def start(self):
        
        self.utasks=[]
        for task in self.tasks:
            if not self.ts.done(task):
                self.utasks.append(task)
        self.ts.save_state()
        
        for task in self.utasks:
            self.run_single(task)
            time.sleep(self.waiting_time)
        
        self.pool.close()
        self.pool.join()
    
    def post_process(self,fun):
        for task in self.tasks:
            dirname=self.generate_directory(task)
            os.chdir(dirname)
            self.process_res.append(fun(**task))
            os.chdir(self.current_directory)
    
    def get_res(self,key=None,value=None):
        res=[]
        par=[]
        
        if len(self.parameters)==1:
            for i,task in enumerate(self.tasks):
                par.append(task.values()[0])
                res.append(self.process_res[i])
        elif len(self.parameters)==2:
            for i,task in enumerate(self.tasks):
                if key in task and task[key]==value:
                    res.append(self.process_res[i])
                    tmp_task=dict(task)
                    del tmp_task[key]
                    par.append(tmp_task.values()[0])
        else:
            raise NotImplementedError('Only support one- and two- parameter case!')
        
        return np.array(par),np.array(res)
    

    
def task(p1,p2):
    print 'current directory:',os.getcwd()
    res='p1='+str(p1)+'  p2='+str(p2)
    res= 1/0
    with open('res.txt','w') as f:
        f.write(res)
    time.sleep(3)


if __name__=="__main__":
    tasks=BatchTasks(task,4)
    tasks.add_parameters('p1',['a','b','c'])
    tasks.add_parameters('p2',range(1,5))
    tasks.start()
    